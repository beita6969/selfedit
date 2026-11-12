"""PY operator read harness (selfedit v5 pilot, coordinator 11-11): the model may run its OWN Python on the row's image(s) during a read.

One PY read of (row, sample):
  turn 1   the general base request (v5t000 = image floor min_image_tokens 4096, no procedures; reasoning effort medium; the row's prompt
           and images exactly as engine_v4.common.content_for builds them) + ONE task-free interface note (NOTE, a system message,
           identical for every row).
  if the reply ends with a fenced python block: the block runs in sandbox.py on the row's ORIGINAL image files; printed text (<= 2,000
           characters) and up to 2 show() images (sent under the same floor rule) come back as the next user turn; the model continues.
  at most K = 2 code rounds; a reply without a block is the final answer (graded with the project grader on that reply).
Only rows with v4_role show / monitor of the clean learn file are ever read (JUDGE rows raise).

The client is CPU only. It talks to the stock E servers of V4/engines.json (state "up") with at most --per-server requests in flight per
server, and starts, stops or changes nothing there. Output (append-only): OUT/reads.jsonl (full traces), OUT/compact.jsonl (no reasoning).

usage: py_read.py --name NAME [--per-bench N] [--samples 0,1] [--per-server 8] [--rows FILE] [--engines FILE] [--out DIR]
"""
import argparse
import base64
import hashlib
import http.client
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _d in (os.environ.get("ENGINE_V4"), "/workspace/fudan/mingdazhang/selfedit-v4/engine_v4", os.path.join(HERE, "..", "engine_v4")):
    if _d and os.path.exists(os.path.join(_d, "common.py")):
        sys.path.insert(0, _d)
        break
import common            # engine_v4: content_for, grade, smart_resize, SAMPLING, EFFORT (the dispatcher's own request pieces)
import sandbox

LEARN_SHA = "b1263f834ed1d2db1c1d55442556118a29853e667f342991d2befb346d14e50e"
SIGNAL_ROLES = ("show", "monitor")
ALLOWED_ROLES = list(SIGNAL_ROLES)      # the roles this run may read: show / monitor unless --roles judge is given (the coordinator's one
                                        # JUDGE-row test of the escalation policy, 11-11 21:20); ID-test and OOD rows are not in the file
_MIT = os.environ.get("PY_MIN_IMAGE_TOKENS", "4096").strip().lower()
BASE_SETTINGS = {"min_image_tokens": None if _MIT in ("", "0", "none", "stock") else int(_MIT), "max_image_tokens": None}
# default = v5t000 (request floor 4096). "none" = STOCK requests, for a weights system that carries its own preprocessor bounds (VP2M512:
# 2,048-4,096 merged image tokens per image on the server side); the client then counts a show() image at PY_SHOWN_TOKENS for its fit test.
SHOWN_TOKENS = int(os.environ.get("PY_SHOWN_TOKENS", "4096"))
MODEL = os.environ.get("PY_MODEL", "stock")
K_ROUNDS = 2
READ_MAX, MAXLEN, GEN_ROOM = 32768, int(os.environ.get("PY_MAXLEN", "46000")), 6000      # MAXLEN = the servers' context window (46,000 in full1)
THINK_ROOM = int(os.environ.get("PY_THINK_ROOM", "6000"))   # full1 ran with 6000; 16000 is the setting proposed after full1
CRN_SEED = 101

def make_note(k):
    return (
        "You can run Python code during this conversation. To run code, end your reply with one fenced block:\n"
        "```python\n<code>\n```\n"
        "The block is executed and what it prints is sent back to you in the next message; then you continue. "
        "A reply that does not end with such a block is your final answer. Use only this fenced form (there is no other call syntax).\n"
        "In the code, the images of this conversation are already loaded: `imgs` is the list of them, in order, as PIL RGB images at their "
        "original pixel size (`img` is `imgs[0]`; get sizes from `.size`, they can differ from the size at which you see the images). "
        "There are no image files: nothing can be opened or saved. Already imported: " + sandbox.available_text() + "; nothing else is "
        "installed (no matplotlib). `print(...)` returns text to you; `show(x)` returns an image to you (a PIL image or an array; at most "
        + str(sandbox.MAX_SHOW) + " per run).\n"
        "Limits: at most " + str(k) + " runs per question (variables are kept from one run to the next); "
        + str(sandbox.CPU_S) + " seconds of CPU time; printed text is cut at 2,000 characters; no file or network access."
    )


NOTE = make_note(K_ROUNDS)
NOTE_SHA = hashlib.sha256(NOTE.encode()).hexdigest()[:16]


HINT = "Two earlier attempts at this question ended with different answers: {a} and {b}. Decide what the picture supports; your answer may also be neither."


def crn_seed(row, sample):
    return zlib.crc32(f"{CRN_SEED}|0|{row}".encode()) % (2 ** 31 - 1024) + int(sample)


# ---------------------------------------------------------------------------------------------------- code block at the end of a reply
_OPEN = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\r?\n", re.I)
_TAGLINE = re.compile(r"^\s*</?\s*(?:tool_call|tool_response|function|parameter|python|invoke|code|tool)\b[^>\n]*>?\s*$", re.I)


def _fenced(c):
    ms = list(_OPEN.finditer(c))
    if not ms:
        return None
    m = ms[-1]
    rest = c[m.end():]
    j = rest.find("```")
    code, trail = (rest, "") if j < 0 else (rest[:j], rest[j + 3:])
    return m.start(), code, trail


def _tool_call(c):
    """a reply that falls into the model's tool-call syntax instead of the fenced form (smoke 11-11: `<tool_call>\n<python>\n...`):
    the code is what is left of the last <tool_call> body after its tag lines are removed."""
    i = c.rfind("<tool_call>")
    if i < 0:
        return None
    rest = c[i + len("<tool_call>"):]
    j = rest.find("</tool_call>")
    body, trail = (rest, "") if j < 0 else (rest[:j], rest[j + len("</tool_call>"):])
    try:
        obj = json.loads(body.strip())
        arg = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(arg, str):
            arg = json.loads(arg)
        if isinstance(arg.get("code"), str):
            return i, arg["code"], trail
    except (ValueError, AttributeError, TypeError):
        pass
    lines = [ln for ln in body.splitlines() if not _TAGLINE.match(ln) and ln.strip().lower() not in ("python", "py", "python3", "function", "code")]
    code = "\n".join(lines)
    f = _fenced(code)
    if f is not None:
        code = f[1]
    return i, code, trail


def final_block(content):
    """-> (code, trailing text, form) when the reply ENDS with a code block (only short text without a boxed answer may follow it), else
    None. form "fenced" = the documented ```python block (an unclosed last fence counts: the reply stopped inside the block);
    form "tool_call" = the fallback above."""
    c = content or ""
    f, t = _fenced(c), _tool_call(c)
    if t is not None and not t[1].strip():        # an empty <tool_call> stub (after full1): ignore it, cut the reply there
        c = c[:t[0]]
        f, t = _fenced(c), _tool_call(c)
        if t is not None and not t[1].strip():
            t = None
    if f is not None and not f[1].strip():        # an empty last fence: fall back to the block before it
        c2 = c[:f[0]]
        f = _fenced(c2)
        if f is not None and len(c2[f[0]:]) - len(f[1]) > 400:
            f = None
    if f is not None and t is not None and t[0] < f[0] and "```" in c[t[0]:]:
        cand, form = (t if "<tool_call>" in c[:f[0]] and "</tool_call>" not in c[t[0]:f[0]] else f), None
        form = "tool_call" if cand is t else "fenced"
    elif f is not None and (t is None or f[0] > t[0]):
        cand, form = f, "fenced"
    elif t is not None:
        cand, form = t, "tool_call"
    else:
        return None
    _, code, trail = cand
    trail = "\n".join(ln for ln in trail.splitlines() if not _TAGLINE.match(ln)).strip()
    if "\\boxed" in trail or "\\fbox" in trail or len(trail) > 300:
        return None
    if not code.strip():
        return None
    return code.strip("\n") + "\n", trail, form


# ---------------------------------------------------------------------------------------------------- images under the floor rule
_enc = threading.Semaphore(4)
_sbx = threading.Semaphore(int(os.environ.get("PY_SANDBOX_PAR", "4")))


def shown_part(png, min_tokens):
    """a show() image -> (image_url part, visual tokens) with the same floor as the row's images; None when it cannot be sent."""
    from PIL import Image
    with _enc:
        im = Image.open(io.BytesIO(png))
        im.load()
        im = im.convert("RGB")
        w, h = im.size
        mn, mx = common.view_pixels(min_tokens, None)
        try:
            hb, wb = common.smart_resize(h, w, min_pixels=mn, max_pixels=mx)
        except ValueError:
            return None
        if (hb, wb) != (h, w):
            im = im.resize((wb, hb), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="PNG", compress_level=1)
    return ({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            (hb // common.FACTOR) * (wb // common.FACTOR) if min_tokens else SHOWN_TOKENS)


def base_content(row):
    with _enc:
        return common.content_for(row["prompt"], row.get("image_paths") or [], BASE_SETTINGS["min_image_tokens"],
                                  BASE_SETTINGS["max_image_tokens"], None)


NO_RUNS_TEXT = "[python: no runs left. The last block was not run.]\nGive your final answer now."


def run_text(j, res, n_sent, n_dropped):
    """the text of the user turn that returns run j (1-based) to the model: fixed layout, no task content."""
    head = f"[python run {j} of {K_ROUNDS}]"
    if res["status"] == "refused":
        body = "The block was not run: " + (res["error"] or "not allowed")
    else:
        out = res["stdout"].rstrip("\n")
        body = "Printed output:\n" + (out if out.strip() else "(nothing was printed)")
        if res["error"]:
            body += "\nError: " + res["error"]
    tail = []
    if n_sent:
        tail.append(f"[{n_sent} image{'s' if n_sent > 1 else ''} from show() follow{'s' if n_sent == 1 else ''}]")
    if n_dropped:
        tail.append(f"[{n_dropped} image{'s' if n_dropped > 1 else ''} from show() could not be sent]")
    if j >= K_ROUNDS:
        tail.append("[no runs left]")
    return head + "\n" + body + ("\n" + "\n".join(tail) if tail else "")


# ---------------------------------------------------------------------------------------------------- servers
class Pool:
    """E servers of engines.json in state "up"; at most `cap` of OUR requests in flight per server."""

    def __init__(self, path, cap, servers=None, cap_file=None, cap_max=None):
        self.path, self.cap = path, cap
        self.cap_file, self.cap_max, self.t_cap = cap_file, cap_max or cap, 0.0
        self.cv = threading.Condition()
        self.inflight, self.bad, self.t_read = {}, {}, 0.0
        self.fixed = servers
        self.servers = []
        self.refresh(force=True)

    def refresh(self, force=False):
        if self.cap_file and time.time() - self.t_cap > 10:        # the cap can be raised / lowered while running: an integer in a file
            self.t_cap = time.time()
            try:
                c = int(open(self.cap_file).read().strip())
                if 1 <= c <= self.cap_max and c != self.cap:
                    print(f"{time.strftime('%H:%M:%S')} per-server cap {self.cap} -> {c}", flush=True)
                    self.cap = c
            except (OSError, ValueError):
                pass
        if self.fixed is not None:
            self.servers = list(self.fixed)
            return
        if not force and time.time() - self.t_read < 30:
            return
        self.t_read = time.time()
        try:
            cfg = json.load(open(self.path))
        except Exception:
            return
        out = []
        for s in cfg.get("servers", []):
            roles = s.get("roles") or [s.get("role", "E")]
            if "E" in roles and s.get("state", "up") == "up" and s.get("weights", "stock") == "stock" and not s.get("edit_sha"):
                out.append(f"{s.get('host', '127.0.0.1')}:{s['port']}")
        self.servers = out

    def acquire(self, prefer=None):
        with self.cv:
            while True:
                self.refresh()
                now = time.time()
                free = [s for s in self.servers if self.inflight.get(s, 0) < self.cap and self.bad.get(s, 0) < now]
                if free:
                    s = prefer if prefer in free else min(free, key=lambda x: self.inflight.get(x, 0))
                    self.inflight[s] = self.inflight.get(s, 0) + 1
                    return s
                self.cv.wait(timeout=5)

    def release(self, s, bad_for=0.0):
        with self.cv:
            self.inflight[s] = self.inflight.get(s, 0) - 1
            if bad_for:
                self.bad[s] = time.time() + bad_for
            self.cv.notify_all()


class CtxError(RuntimeError):
    pass


def post(pool, body, prefer=None, tries=3, timeout=7200):
    """one chat call -> (json, server). Context overflow raises CtxError; a refused connection moves to another server."""
    err = None
    for k in range(tries + 3):
        srv = pool.acquire(prefer)
        bad = 0.0
        try:
            host, port = srv.rsplit(":", 1)
            conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
            try:
                conn.request("POST", "/v1/chat/completions", body=json.dumps(dict(body, model=MODEL)).encode(),
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                txt = resp.read().decode("utf-8", "replace")
            finally:
                conn.close()
            if resp.status == 200:
                return json.loads(txt), srv
            err = f"HTTP {resp.status}: {txt[:300]}"
            if "maximum context length" in txt or "longer than the maximum" in txt:
                raise CtxError(err)
        except CtxError:
            raise
        except (ConnectionError, OSError) as e:
            err, bad, prefer = f"{type(e).__name__}: {str(e)[:200]}", 60.0, None
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            pool.release(srv, bad)
        time.sleep(min(60, 5 * 3 ** k))
    raise RuntimeError(err or "request failed")


# ---------------------------------------------------------------------------------------------------- one read
def est_tokens(text):
    return len(text) // 3 + 8


def one_read(pool, name, row, sample, keep_think=True, task=None):
    assert row.get("v4_role") in ALLOWED_ROLES, f"row {row['id']} has role {row.get('v4_role')}: this run reads only {ALLOWED_ROLES}"
    t0 = time.time()
    seed = crn_seed(row["id"], sample)
    rec = dict(key=f"py|{name}|{row['id']}|{sample}", name=name, row=row["id"], sample=sample, bench=row["bench"], role=row["v4_role"],
               kind=row["kind"], answer=row["answer"], seed=seed, note_sha=NOTE_SHA, settings=BASE_SETTINGS, keep_think=keep_think,
               calls=[], runs=[], flags={}, t_start=t0)
    content0, ntok0 = base_content(row)
    rec["image_tokens"] = ntok0
    if task is not None:
        rec["rep"], rec["hint"] = task.get("rep"), task.get("hint")
        if task.get("hint"):                 # PYA: the user turn ends with the two disputed answers (verbatim, order fixed by the task file)
            content0 = list(content0) + [{"type": "text", "text": "\n\n" + HINT.format(a=task["hint"][0], b=task["hint"][1])}]
    raw_images = [open(p, "rb").read() for p in (row.get("image_paths") or [])]
    msgs = [{"role": "system", "content": NOTE}, {"role": "user", "content": content0}]
    body0 = dict(max_tokens=READ_MAX, chat_template_kwargs={"reasoning_effort": common.EFFORT, "enable_thinking": True}, seed=seed,
                 cache_salt=hashlib.sha256(("py|" + rec["key"]).encode()).hexdigest()[:32], **common.SAMPLING)
    srv, final, finish = None, "", None
    think_tokens = 0                       # reasoning tokens of earlier assistant turns kept in the history
    while True:
        try:
            o, srv = post(pool, dict(body0, messages=msgs), prefer=srv)
        except CtxError as e:
            # history too long: first drop the reasoning of earlier turns, then give up on the read's continuation
            if keep_think and any(m.get("reasoning") for m in msgs):
                for m in msgs:
                    m.pop("reasoning", None)
                rec["flags"]["ctx_drop_think"] = True
                think_tokens = 0
                continue
            rec["flags"]["ctx_stop"] = str(e)[:200]
            break
        c = o["choices"][0]
        m = c["message"]
        content = (m.get("content") or "").strip()
        reasoning = (m.get("reasoning_content") or m.get("reasoning") or "").strip()
        finish = c.get("finish_reason")
        u = o["usage"]
        rec["calls"].append(dict(prompt_tokens=u["prompt_tokens"], tokens=u["completion_tokens"], finish=finish, server=srv,
                                 t=round(time.time() - t0, 1), content=content, reasoning=reasoning))
        final = content
        blk = final_block(content) if finish == "stop" else None
        if blk is None:
            break
        if len(rec["runs"]) >= K_ROUNDS:
            # no runs left and the reply is another block: say so once (nothing is run) and take the next reply as the answer
            rec["flags"]["code_after_limit"] = rec["flags"].get("code_after_limit", 0) + 1
            if rec["flags"]["code_after_limit"] > 1 or u["prompt_tokens"] + u["completion_tokens"] + 1024 > MAXLEN:
                break
            am = {"role": "assistant", "content": content}
            if keep_think and reasoning and not rec["flags"].get("ctx_drop_think"):
                am["reasoning"] = reasoning
            msgs = msgs + [am, {"role": "user", "content": [{"type": "text", "text": NO_RUNS_TEXT}]}]
            continue
        code, trail, form = blk
        if form != "fenced":
            rec["flags"]["tool_call_form"] = rec["flags"].get("tool_call_form", 0) + 1
        with _sbx:      # notebook state: the read's earlier blocks that ran (ok, or failed midway) are re-run silently before this one
            res = sandbox.run(code, raw_images, prelude=[r["code"] for r in rec["runs"] if r["status"] in ("ok", "error")])
        j = len(rec["runs"]) + 1
        # ---- what fits: prompt of the next call = this call's prompt + its output (reasoning kept or not) + the run's text and images
        keep_now = keep_think
        hist = u["prompt_tokens"] + (u["completion_tokens"] if keep_now else est_tokens(content))
        parts, toks = [], []
        for png in res["images"]:
            p = shown_part(png, BASE_SETTINGS["min_image_tokens"])
            if p is not None:
                parts.append(p[0])
                toks.append(p[1])
        n_made = len(parts)
        text_est = est_tokens(res["stdout"]) + est_tokens(res["error"] or "") + 60
        while parts and hist + text_est + sum(toks) + GEN_ROOM > MAXLEN:
            parts.pop()
            toks.pop()
        if hist + text_est + sum(toks) + max(GEN_ROOM, THINK_ROOM) > MAXLEN and keep_now:
            est_reason = max(0, u["completion_tokens"] - est_tokens(content)) + think_tokens
            if hist - est_reason + text_est + GEN_ROOM <= MAXLEN:
                keep_now = False
                for mm in msgs:
                    mm.pop("reasoning", None)
                think_tokens = 0
                rec["flags"]["ctx_drop_think"] = True
                hist -= est_reason
        n_dropped = len(res["images"]) - len(parts)
        text = run_text(j, res, len(parts), n_dropped)
        rec["runs"].append(dict(code=code, trail=trail[:300], form=form, status=res["status"], error=res["error"], stdout=res["stdout"],
                                n_images=len(res["images"]), n_sent=len(parts), image_tokens=toks, wall=round(res["wall"], 2),
                                cpu=res.get("cpu"), lock=res.get("lock"), text=text))
        if n_dropped and n_made > len(parts):
            rec["flags"]["ctx_drop_images"] = rec["flags"].get("ctx_drop_images", 0) + (n_made - len(parts))
        if hist + text_est + sum(toks) + 1024 > MAXLEN:
            rec["flags"]["ctx_stop"] = "history does not fit"
            break
        am = {"role": "assistant", "content": content}
        if keep_now and reasoning:
            am["reasoning"] = reasoning
            think_tokens += max(0, u["completion_tokens"] - est_tokens(content))
        msgs = msgs + [am, {"role": "user", "content": [{"type": "text", "text": text}] + parts}]
    tokens = sum(x["tokens"] for x in rec["calls"])
    ok, loop, gerr, pred = common.grade(row, final, rec["calls"][-1]["reasoning"] if rec["calls"] else "", finish, tokens, sample)
    if rec["flags"].get("code_after_limit") or rec["flags"].get("ctx_stop"):
        pass                                # graded as it stands: the last reply is the answer (no answer -> wrong)
    rec.update(final=final, finish=finish, ok=bool(ok), loop=loop, grade_error=gerr, pred=pred, tokens=tokens,
               prompt_tokens=sum(x["prompt_tokens"] for x in rec["calls"]), n_calls=len(rec["calls"]), n_runs=len(rec["runs"]),
               options=row.get("options") or [], category=row.get("category"), t_done=time.time())
    return rec


def compact(rec):
    c = {k: rec.get(k) for k in ("key", "row", "sample", "bench", "role", "kind", "answer", "ok", "pred", "finish", "tokens", "prompt_tokens",
                                  "n_calls", "n_runs", "flags", "error", "image_tokens", "loop", "rep", "hint")}
    c["calls"] = [dict(prompt_tokens=x["prompt_tokens"], tokens=x["tokens"], finish=x["finish"], server=x["server"], t=x["t"],
                       n_reason=len(x["reasoning"]), n_content=len(x["content"])) for x in rec.get("calls", [])]
    c["runs"] = [dict(code=r["code"][:3000], status=r["status"], error=(r["error"] or "")[:400] or None, stdout=r["stdout"][:800],
                      n_images=r["n_images"], n_sent=r["n_sent"], wall=r["wall"], cpu=r["cpu"]) for r in rec.get("runs", [])]
    c["t"] = round((rec.get("t_done") or time.time()) - rec["t_start"], 1)
    return c


# ---------------------------------------------------------------------------------------------------- driver
GOLD_FIX = {"babyvision/6557": "="}     # held-out: the Vesta copy has an empty gold for this row; the Venus file says "=" (coordinator 11-11 22:20)


def load_heldout(paths):
    """REPORTING ONLY (coordinator 11-11 22:20): the held-out row files (ID test, OOD v2); every row gets the role "heldout"."""
    out = []
    for path in paths.split(","):
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                r["v4_role"] = "heldout"
                if r["id"] in GOLD_FIX and r.get("answer") in ("", None):
                    r["answer"] = GOLD_FIX[r["id"]]
                out.append(r)
    if len({r["id"] for r in out}) != len(out):
        raise SystemExit("held-out row ids are not unique")
    return out


def load_rows(path, check_sha=True):
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if check_sha and sha != LEARN_SHA:
        raise SystemExit(f"{path}: sha256 {sha} is not the clean 768-row learn file")
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    sig = [r for r in rows if r.get("v4_role") in ALLOWED_ROLES]
    return sig


def pick(sig, per_bench):
    if not per_bench:
        return sig
    out = []
    for b in sorted({r["bench"] for r in sig}):
        rs = sorted((r for r in sig if r["bench"] == b), key=lambda r: zlib.crc32(("pick|" + r["id"]).encode()))
        out += rs[:per_bench]
    return out


def main():
    global K_ROUNDS, NOTE, NOTE_SHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rows", default="/data/fd/mingdazhang/selfedit/v4/rows/learn.jsonl")
    ap.add_argument("--engines", default="/data/fd/mingdazhang/selfedit/v4/engines.json")
    ap.add_argument("--servers", default=None, help="host:port,... instead of engines.json (tests)")
    ap.add_argument("--out", default="/data/fd/mingdazhang/selfedit/v5/py_op")
    ap.add_argument("--per-bench", type=int, default=0)
    ap.add_argument("--samples", default="0")
    ap.add_argument("--per-server", type=int, default=8)
    ap.add_argument("--no-think", action="store_true", help="do not keep the reasoning of earlier turns in the history")
    ap.add_argument("--no-sha", action="store_true", help="tests only: rows file is not the learn file")
    ap.add_argument("--k", type=int, default=K_ROUNDS, help="code rounds per read (the pilot's setting is 2)")
    ap.add_argument("--roles", default="show,monitor", help="show,monitor (default) or judge")
    ap.add_argument("--row-ids", default=None, help="JSON list of row ids: read only these rows (they must have an allowed role)")
    ap.add_argument("--tasks", default=None, help="JSON list of {row, sample, rep, hint: [a, b] | null}: exactly these reads (generation 2)")
    ap.add_argument("--heldout", action="store_true", help="REPORTING ONLY: --rows = held-out row file(s), comma-separated; needs --row-ids")
    ap.add_argument("--cap-file", default=None, help="a file holding the per-server cap (integer), re-read while running")
    ap.add_argument("--max-per-server", type=int, default=0, help="upper bound of the cap file (sizes the worker pool)")
    a = ap.parse_args()
    roles = [x for x in a.roles.split(",") if x]
    if not set(roles) <= {"show", "monitor", "judge"} or ("judge" in roles and len(roles) > 1):
        raise SystemExit("--roles: show,monitor or judge")
    ALLOWED_ROLES[:] = roles
    if a.heldout:
        if not a.row_ids and not a.tasks:
            raise SystemExit("--heldout needs --row-ids or --tasks")
        ALLOWED_ROLES[:] = ["heldout"]
    K_ROUNDS, NOTE = a.k, make_note(a.k)
    NOTE_SHA = hashlib.sha256(NOTE.encode()).hexdigest()[:16]
    sig = load_heldout(a.rows) if a.heldout else load_rows(a.rows, check_sha=not a.no_sha)
    if a.row_ids:
        want = set(json.load(open(a.row_ids)))
        have = {r["id"] for r in sig}
        if not want <= have:
            raise SystemExit(f"--row-ids: {len(want - have)} ids are not rows of role {ALLOWED_ROLES}")
        sig = [r for r in sig if r["id"] in want]
    sig = pick(sig, a.per_bench)
    assert all(r["v4_role"] in ALLOWED_ROLES for r in sig)
    samples = [int(s) for s in a.samples.split(",")]
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    f_full, f_comp = os.path.join(out, "reads.jsonl"), os.path.join(out, "compact.jsonl")
    done = set()
    if os.path.exists(f_full):
        for l in open(f_full):
            try:
                x = json.loads(l)
            except ValueError:
                continue
            if not x.get("error"):
                done.add((x["row"], x["sample"]))
    todo = [(r, s, None) for s in samples for r in sig if (r["id"], s) not in done]
    if a.tasks:
        byid = {r["id"]: r for r in sig}
        T = json.load(open(a.tasks))
        if not all(t["row"] in byid for t in T):
            raise SystemExit(f"--tasks: rows that are not of role {ALLOWED_ROLES}")
        todo = [(byid[t["row"]], int(t["sample"]), t) for t in sorted(T, key=lambda t: (int(t["sample"]), t["row"])) if (t["row"], int(t["sample"])) not in done]
    cap_max = max(a.per_server, a.max_per_server)
    pool = Pool(a.engines, a.per_server, servers=a.servers.split(",") if a.servers else None, cap_file=a.cap_file, cap_max=cap_max)
    json.dump(dict(name=a.name, note=NOTE, note_sha=NOTE_SHA, k=K_ROUNDS, settings=BASE_SETTINGS, effort=common.EFFORT, sampling=common.SAMPLING,
                   rows=len(sig), samples=samples, per_server=a.per_server, keep_think=not a.no_think, servers=pool.servers, roles=ALLOWED_ROLES, think_room=THINK_ROOM, maxlen=MAXLEN, model=MODEL, shown_tokens=SHOWN_TOKENS,
                   available=sandbox.available_text(), t=time.time()), open(os.path.join(out, f"config_{int(time.time())}.json"), "w"), indent=1)
    print(f"{a.name}: rows {len(sig)} x samples {samples}; todo {len(todo)} (done {len(done)}); servers {pool.servers} cap {a.per_server}; "
          f"note sha {NOTE_SHA}; jail {sandbox.jail_dir()}", flush=True)
    if not pool.servers:
        raise SystemExit("no E server in state up")
    lock = threading.Lock()
    stat = dict(n=0, ok=0, code=0, err=0, t0=time.time())

    def work(item):
        row, s, task = item
        try:
            rec = one_read(pool, a.name, row, s, keep_think=not a.no_think, task=task)
        except Exception as e:
            rec = dict(key=f"py|{a.name}|{row['id']}|{s}", row=row["id"], sample=s, bench=row["bench"], role=row["v4_role"],
                       error=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[-1500:], t_start=time.time(), calls=[], runs=[])
        with lock:
            with open(f_full, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(f_comp, "a") as f:
                f.write(json.dumps(compact(rec), ensure_ascii=False) + "\n")
            stat["n"] += 1
            stat["ok"] += bool(rec.get("ok"))
            stat["code"] += bool(rec.get("runs"))
            stat["err"] += bool(rec.get("error"))
            if stat["n"] % 10 == 0 or stat["n"] == len(todo):
                print(f"{time.strftime('%H:%M:%S')} {stat['n']}/{len(todo)} ok {stat['ok']} ran-code {stat['code']} errors {stat['err']} "
                      f"elapsed {time.time() - stat['t0']:.0f}s inflight {dict(pool.inflight)}", flush=True)

    n_workers = max(1, min(len(todo), cap_max * max(1, len(pool.servers))))
    with ThreadPoolExecutor(n_workers) as ex:
        list(ex.map(work, todo))
    print(f"{a.name}: finished {stat['n']} reads, ok {stat['ok']}, ran code {stat['code']}, errors {stat['err']}, "
          f"{time.time() - stat['t0']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
