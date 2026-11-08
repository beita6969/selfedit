"""Pipeline precheck: is this model, at this budget, in the state where a stop edit can pay?

The rare successes share a blocked capability: traces that hold the answer run out of budget, and answers that finish are right.
Reads two base trace files of the same problems, a tight and a loose budget, and generates nothing.

Usage: python rsi/precheck.py --tight TIGHT.jsonl --loose LOOSE.jsonl --budget 8192 --out OUT.md [--json OUT.json]
"""
import argparse, json, re

ap = argparse.ArgumentParser()
ap.add_argument("--tight", required=True)
ap.add_argument("--loose", required=True)
ap.add_argument("--budget", type=int, default=8192)
ap.add_argument("--out", required=True)
ap.add_argument("--json", default=None)
args = ap.parse_args()

CONCL = re.compile(r"(?i)(final answer|the answer is|answer is|so the answer|therefore,? the|thus,? the|answer:|\\boxed)")


def load(p):
    return [json.loads(l) for l in open(p)]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def mention(r):
    """Fraction of the thinking text before the correct answer's first mention; None if never or the answer is too short to search."""
    a, t = str(r["answer"]).strip(), r["reasoning"] or ""
    if len(a) < 2 or not a.isdigit() or not t:
        return None
    m = re.search(rf"(?<!\d){re.escape(a)}(?!\d)", t)
    return m.start() / len(t) if m else None


def summary(R):
    fin = [r for r in R if r["finish"] != "length"]
    return {"traces": len(R), "acc": rate(r["ok"] for r in R), "trunc": 1 - len(fin) / max(1, len(R)), "fin_correct": rate(r["ok"] for r in fin),
            "tokens": rate(r["tokens"] for r in R)}


T, L = load(args.tight), load(args.loose)
st, sl = summary(T), summary(L)
right = [r for r in L if r["ok"] and r["finish"] != "length"]
tok = [r["tokens"] for r in right]

# Budget sensitivity: accuracy at the tight budget if every token count of the loose traces were multiplied by s.
sens = {s: rate(r["ok"] and r["finish"] != "length" and r["tokens"] * s <= args.budget for r in L) for s in (1.0, .9, .8, .7, .6)}

# Where the answer first appears in right traces, and how much is spent after it.
pos = [p for p in (mention(r) for r in right) if p is not None]
after = [(1 - p) for p in pos]

# Right loose traces cut at the tight budget: did the answer appear before the cut? Closing there is the most a selective stop can win.
long_right = [r for r in right if r["tokens"] > args.budget]
inreach = []
for r in long_right:
    p = mention(r)
    if p is not None:
        inreach.append(p * r["tokens"] <= args.budget)
ceil_mention = rate(inreach) * len(long_right) / max(1, len(L))

# Truncated tight traces: how many have concluded, and how many state the correct answer on a concluding line.
cut = [r for r in T if r["finish"] == "length"]
n_concl = n_right = 0
for r in cut:
    ls = [l for l in (r["reasoning"] or "").split("\n") if CONCL.search(l)]
    n_concl += bool(ls)
    a = str(r["answer"]).strip()
    n_right += any(re.search(rf"(?<!\d){re.escape(a)}(?!\d)", l) for l in ls)

out = {"tight": st, "loose": sl, "right_tokens": {"q10": q(tok, .1), "q50": q(tok, .5), "q90": q(tok, .9)}, "sensitivity": sens,
       "first_mention": {"n": len(pos), "q25": q(pos, .25), "q50": q(pos, .5), "q75": q(pos, .75)}, "after_share_mean": rate(after),
       "long_right": len(long_right), "long_right_searchable": len(inreach), "long_right_in_reach": rate(inreach), "ceiling_at_mention": ceil_mention,
       "cut": len(cut), "cut_concluded": n_concl, "cut_states_answer": n_right, "ceiling_at_conclusion": n_right / max(1, len(T))}
gain = sens[.8] - sens[1.0]
blocked = "no" if st["trunc"] < .3 or st["fin_correct"] < .8 or ceil_mention < .05 else "strong" if ceil_mention >= .25 else "partial"
out["gain_per_20pct_tokens"], out["blocked"] = gain, blocked

w = f"""# Precheck: your state at this budget, from your own traces, nothing generated
Tight budget {args.budget}: accuracy {st['acc']:.1%}, truncated {st['trunc']:.1%}, answers that finish are right {st['fin_correct']:.1%}, mean tokens {st['tokens']:.0f}.
Loose budget: accuracy {sl['acc']:.1%}, truncated {sl['trunc']:.1%}, answers that finish are right {sl['fin_correct']:.1%}, mean tokens {sl['tokens']:.0f}.
Tokens of right traces at the loose budget: q10 {q(tok, .1)}, median {q(tok, .5)}, q90 {q(tok, .9)}.
Accuracy at the tight budget if all tokens scaled by s: """ + ", ".join(f"{s:.1f}: {v:.1%}" for s, v in sens.items()) + f""". A 20 % saving is worth {gain:+.1%}.
The correct answer first appears at {q(pos, .5) if pos else float('nan'):.0%} of a right trace (median, q25 {q(pos, .25) if pos else float('nan'):.0%}, q75 {q(pos, .75) if pos else float('nan'):.0%}, {len(pos)} searchable).
Right traces cut at the tight budget: {len(long_right)}; the answer appears before the cut in {rate(inreach):.0%} of the {len(inreach)} searchable. Closing there is worth at most {ceil_mention:+.1%}.
Truncated traces at the tight budget: {len(cut)}; with a concluding line {n_concl}; stating the correct answer there {n_right}, worth at most {n_right / max(1, len(T)):+.1%}.
Blocked capability: {dict(strong="strong, a selective stop edit can pay a lot", partial="partial, a selective stop edit can pay a little; most traces are still working toward the answer when the budget ends", no="none, the budget is not what stands between these traces and the answer")[blocked]}.
"""
open(args.out, "w").write(w)
if args.json:
    json.dump(out, open(args.json, "w"), indent=1)
print(w)
