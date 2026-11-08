"""Sample n answers per multimodal item from an OpenAI-compatible vLLM server; append one JSON line per sample (resumable).

Images are sent as base64 data URLs. A prompt containing exactly len(image_paths) "<image>" placeholders is interleaved
(text, image, text, ...); otherwise all images go first, then the text. Records carry set = "<dir>/<stem>" (sets/blink vs
pool/blink). --limit/--limits take the first N lines of a set (prep_mm.py writes stratified order, so that is a stratified
subset). Without --interleave items run set by set; with it all sets of a phase advance at the same fraction of their
limit, so a --deadline-min cut leaves every set with a stratified prefix done. --sets2 (e.g. pools) start after --sets.
"""
import argparse, asyncio, base64, json, mimetypes, os, time
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError

ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+", required=True)          # JSONL files from prep_mm.py
ap.add_argument("--out", required=True)
ap.add_argument("--servers", nargs="+", required=True)       # e.g. http://127.0.0.1:8000/v1
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=0)          # 0 = let vLLM use max_model_len - prompt_len
ap.add_argument("--effort", default=None)                     # Qwen3.8: xhigh / medium / low
ap.add_argument("--thinking", default="on", choices=["on", "off", "none"])
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--presence-penalty", type=float, default=0.0)
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--limit", type=int, default=0)               # first N items per set
ap.add_argument("--limits", default="")                       # per-set overrides, e.g. "sets/vstar=191,pool/blink=60"
ap.add_argument("--sets2", nargs="*", default=[])             # second phase (e.g. pools): started only after all --sets items
ap.add_argument("--interleave", action="store_true")          # within a phase, advance all sets at the same fraction of their limit
ap.add_argument("--deadline-min", type=float, default=0)      # stop starting new requests after this many minutes
ap.add_argument("--hard-deadline-min", type=float, default=0) # cancel in-flight requests (not recorded) after this many minutes
ap.add_argument("--retry-errors", action="store_true")        # on resume, rerun samples that ended in an error record
args = ap.parse_args()

done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        r = json.loads(line)
        if r["finish"] != "error" or not args.retry_errors:
            done.add((r["id"], r["sample"]))
per = dict(kv.split("=") for kv in args.limits.split(",") if kv)


def load_phase(files, phase):
    keyed = []
    for f in files:
        stem = os.path.basename(f).rsplit(".", 1)[0]
        name = f"{os.path.basename(os.path.dirname(os.path.abspath(f)))}/{stem}"  # e.g. sets/blink vs pool/blink
        lim = int(per.get(name, per.get(stem, args.limit)))
        rows = [json.loads(l) for l in open(f)]
        if any(r.get("pair") is not None for r in rows):  # keep pair members adjacent (MMVP) so prefixes hold whole pairs
            first = {}
            for i, r in enumerate(rows):
                first.setdefault(r.get("pair"), i)
            rows.sort(key=lambda r: first[r.get("pair")])
        rows = rows[:lim] if lim else rows
        for i, p in enumerate(rows):
            p["set"] = name
            keyed.append(((i + 0.5) / len(rows) if args.interleave else 0, len(keyed), p))
    return [p for _, _, p in sorted(keyed, key=lambda t: (t[0], t[1]))]


probs = load_phase(args.sets, 0) + load_phase(args.sets2, 1)
jobs = [(p, s) for p in probs for s in range(args.n) if (p["id"], s) not in done]
print(f"{len(done)} done, {len(jobs)} to run over {len(probs)} items", flush=True)

clients = [AsyncOpenAI(base_url=u, api_key="none", timeout=3 * 3600, max_retries=0) for u in args.servers]
sem = asyncio.Semaphore(args.concurrency)
lock = asyncio.Lock()
out = open(args.out, "a")
t_start = time.time()
deadline = t_start + 60 * args.deadline_min if args.deadline_min else None
stats = {"ok": 0, "err": 0, "skipped": 0}


def data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def build_content(prob):
    imgs = [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in prob["image_paths"]]
    parts = prob["prompt"].split("<image>")
    if len(parts) - 1 == len(imgs) and imgs:
        content = []
        for k, txt in enumerate(parts):
            if txt.strip():
                content.append({"type": "text", "text": txt.strip()})
            if k < len(imgs):
                content.append(imgs[k])
        return content
    return imgs + [{"type": "text", "text": prob["prompt"]}]


async def one(k, prob, s):
    extra = {"top_k": args.top_k, "min_p": 0.0, "repetition_penalty": 1.0}
    if args.thinking != "none":
        extra["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    kw = {"reasoning_effort": args.effort} if args.effort else {}
    if args.max_tokens:
        kw["max_tokens"] = args.max_tokens
    async with sem:
        if deadline and time.time() > deadline:
            stats["skipped"] += 1
            return
        msgs = [{"role": "user", "content": build_content(prob)}]
        t0 = time.time()
        rec = {"id": prob["id"], "set": prob["set"], "kind": prob["kind"], "answer": prob["answer"],
               "options": prob.get("options") or [], "category": prob.get("category"), "sample": s}
        if "precision" in prob:
            rec["precision"] = prob["precision"]
        if "pair" in prob:
            rec["pair"] = prob["pair"]
        for attempt in range(4):
            try:
                r = await clients[k % len(clients)].chat.completions.create(
                    model=args.model, messages=msgs, temperature=args.temperature, top_p=args.top_p,
                    presence_penalty=args.presence_penalty, extra_body=extra, **kw)
                m = r.choices[0].message
                rec.update({"reasoning": getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or "",
                            "content": m.content or "", "finish": r.choices[0].finish_reason,
                            "tokens": r.usage.completion_tokens, "prompt_tokens": r.usage.prompt_tokens})
                stats["ok"] += 1
                break
            except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                if attempt == 3:
                    rec.update({"reasoning": "", "content": "", "finish": "error", "tokens": 0, "prompt_tokens": 0,
                                "error": repr(e)[:500]})
                    stats["err"] += 1
                else:
                    await asyncio.sleep(10 * (attempt + 1))
            except Exception as e:  # 4xx (e.g. prompt too long): deterministic, record and move on
                rec.update({"reasoning": "", "content": "", "finish": "error", "tokens": 0, "prompt_tokens": 0,
                            "error": repr(e)[:500]})
                stats["err"] += 1
                break
        rec["secs"] = round(time.time() - t0, 1)
    async with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()


async def main():
    tasks = [asyncio.ensure_future(one(k, p, s)) for k, (p, s) in enumerate(jobs)]
    hard = 60 * args.hard_deadline_min - (time.time() - t_start) if args.hard_deadline_min else None
    try:
        for i, fut in enumerate(asyncio.as_completed(tasks, timeout=hard), 1):
            try:
                await fut
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                print("error:", repr(e)[:300], flush=True)
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(jobs)} {time.strftime('%H:%M:%S')} {stats} {round((time.time() - t_start) / 60, 1)} min", flush=True)
    except asyncio.TimeoutError:
        left = [t for t in tasks if not t.done()]
        print(f"hard deadline: cancelling {len(left)} unfinished requests", flush=True)
        for t in left:
            t.cancel()
        await asyncio.gather(*left, return_exceptions=True)

asyncio.run(main())
print("final", stats, flush=True)
