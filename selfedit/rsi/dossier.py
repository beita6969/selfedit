"""Build the text dossier the model reads about itself: pool performance, failure excerpts, edit history."""
import json, random
from collections import defaultdict


def pool_summary(records, budgets=(8192, 16384, 32768)):
    by = defaultdict(list)
    for r in records:
        by[r["id"].split("/")[0]].append(r)
    lines = []
    for bench, rs in sorted(by.items()):
        n = len(rs)
        acc = sum(r["ok"] for r in rs) / n
        trunc = sum(r["finish"] == "length" for r in rs) / n
        toks = sorted(r["tokens"] for r in rs)
        capped = ", ".join(f"{b // 1024}k {100 * sum(r['ok'] and r['tokens'] <= b for r in rs) / n:.1f}%" for b in budgets)
        lines.append(f"- {bench}: {n} samples, accuracy {100 * acc:.1f}% (if capped at {capped}), "
                     f"tokens median {toks[n // 2]} / p90 {toks[int(n * 0.9)]}, truncated {100 * trunc:.1f}%")
    return "\n".join(lines)


def failure_excerpts(records, k=6, chars=700, seed=0):
    """Up to k excerpts of distinct failed problems: wrong final answers and truncated (often repetitive) tails."""
    rng = random.Random(seed)
    wrong = [r for r in records if not r["ok"] and r["finish"] == "stop"]
    trunc = [r for r in records if r["finish"] == "length"]
    picked, seen = [], set()
    for pool, kind in ((wrong, "wrong answer"), (trunc, "ran out of budget")):
        rng.shuffle(pool)
        for r in pool:
            if r["id"] in seen or len(picked) >= k:
                continue
            seen.add(r["id"])
            text = (r["reasoning"] + "\n</think>\n" + r["content"]) if r["content"] else r["reasoning"]
            picked.append(f"### {r['id']} ({kind}, {r['tokens']} tokens, gold answer {r['answer']})\n...{text[-chars:]}")
    return "\n\n".join(picked)


def edit_history(log_path):
    """One line per evaluated edit: generation, author, op, target, result."""
    lines = []
    try:
        for l in open(log_path):
            e = json.loads(l)
            pre = e.get("pre") or {}
            stop = f", stop read {pre['d_cut']:+.2f}, drift {-pre['d_all']:+.4f}" if "d_cut" in pre else ""
            lines.append(f"- gen {e['gen']} [{e['author']}] {e['op']} {e['target']} {e.get('params', {})}: "
                         f"{e['verdict']} (Δacc {e.get('diff', float('nan')):+.3f}, Δtokens {e.get('dtok', 0):+.0f}{stop})")
    except FileNotFoundError:
        pass
    return "\n".join(lines) or "(no edits yet)"
