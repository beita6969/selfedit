"""Small export of PY reads for reading on the Mac: a stratified sample of traces (or all of them), gzip JSONL.
Classes (base8 = share of the row's 8 cached base reads that are right):
  right_code_base_low   ran code, right, base8 <= 3/8          (candidates: right because of the code)
  wrong_code_base_high  ran code, wrong, base8 >= 5/8          (candidates: wrong because of the code)
  code_same_as_base     ran code, answer = the base's majority answer (code changed nothing)
  run_failed            a run was refused / raised / timed out
  wrong_code_base_low   ran code, wrong, base8 <= 3/8
  direct                answered without code
usage: py_export.py READS.jsonl BASE.jsonl.gz LEARN.jsonl OUT.jsonl.gz [--all | --n 42]"""
import collections
import gzip
import json
import sys

QUOTA = [("right_code_base_low", 12), ("wrong_code_base_high", 12), ("code_same_as_base", 6), ("run_failed", 5), ("wrong_code_base_low", 4), ("direct", 3)]


def cut(t, n, tail=False):
    t = t or ""
    if len(t) <= n:
        return t
    return ("...[cut]... " + t[-n:]) if tail else (t[:n] + " ...[cut]")


def main():
    reads_p, base_p, rows_p, out_p = sys.argv[1:5]
    every = "--all" in sys.argv
    rows = {r["id"]: r for r in map(json.loads, open(rows_p)) if r.get("v4_role") in ("show", "monitor")}
    base = collections.defaultdict(list)
    for l in gzip.open(base_p, "rt"):
        v = json.loads(l)
        if v["row"] in rows:
            base[v["row"]].append(v)
    R = []
    for l in open(reads_p):
        x = json.loads(l)
        if x.get("error") or x.get("ok") is None or x["row"] not in rows:
            continue
        R.append(x)
    R = list({(x["row"], x["sample"]): x for x in R}.values())

    def info(x):
        b = base[x["row"]]
        rate = sum(bool(v["ok"]) for v in b) / max(len(b), 1)
        preds = collections.Counter(str(v.get("pred")) for v in b)
        maj = preds.most_common(1)[0][0] if preds else None
        return rate, preds, maj

    def klass(x):
        rate, preds, maj = info(x)
        if not x.get("n_runs"):
            return ["direct"]
        k = []
        if x["ok"] and rate <= 0.375:
            k.append("right_code_base_low")
        if not x["ok"] and rate >= 0.625:
            k.append("wrong_code_base_high")
        if any(u["status"] != "ok" for u in x["runs"]):
            k.append("run_failed")
        if not x["ok"] and rate <= 0.375:
            k.append("wrong_code_base_low")
        if str(x.get("pred")) == maj:
            k.append("code_same_as_base")
        return k or ["code_other"]

    chosen, seen = [], set()
    if every:
        chosen = [(klass(x)[0], x) for x in sorted(R, key=lambda x: (x["bench"], x["row"], x["sample"]))]
    else:
        for name, q in QUOTA:
            cand = [x for x in R if name in klass(x) and (x["row"], x["sample"]) not in seen]
            byb = collections.defaultdict(list)
            for x in sorted(cand, key=lambda x: (x["row"], x["sample"])):
                byb[x["bench"]].append(x)
            got = 0
            while got < q and any(byb.values()):
                for b in sorted(byb):
                    if byb[b] and got < q:
                        x = byb[b].pop(0)
                        if x["row"] in {c[1]["row"] for c in chosen}:      # one read per row
                            continue
                        chosen.append((name, x))
                        seen.add((x["row"], x["sample"]))
                        got += 1
    counts = collections.Counter(k for x in R for k in klass(x))
    with gzip.open(out_p, "wt") as f:
        f.write(json.dumps(dict(meta=True, reads=len(R), class_counts=dict(counts), chosen=len(chosen))) + "\n")
        for name, x in chosen:
            rate, preds, maj = info(x)
            r = rows[x["row"]]
            f.write(json.dumps(dict(
                klass=name, row=x["row"], sample=x["sample"], bench=x["bench"], category=r.get("category"), kind=r.get("kind"),
                prompt=cut(r["prompt"], 1100), n_images=len(r.get("image_paths") or []), answer=r["answer"], base8=rate, base_preds=dict(preds.most_common(5)),
                ok=x["ok"], pred=x.get("pred"), tokens=x["tokens"], prompt_tokens=x["prompt_tokens"], flags=x.get("flags"), image_tokens=x.get("image_tokens"),
                calls=[dict(tokens=c["tokens"], prompt_tokens=c["prompt_tokens"], finish=c["finish"], reasoning_chars=len(c["reasoning"]),
                            reasoning_tail=cut(c["reasoning"], 1600 if not every else 2200, tail=True), content=cut(c["content"], 3200)) for c in x["calls"]],
                runs=[dict(status=u["status"], error=u["error"], stdout=cut(u["stdout"], 1600), n_images=u["n_images"], n_sent=u["n_sent"],
                           image_tokens=u.get("image_tokens"), cpu=u.get("cpu"), wall=u.get("wall"), lock=u.get("lock") if every else None) for u in x["runs"]],
            ), ensure_ascii=False) + "\n")
    print(f"{out_p}: {len(chosen)} traces of {len(R)} reads; classes {dict(counts)}")


if __name__ == "__main__":
    main()
