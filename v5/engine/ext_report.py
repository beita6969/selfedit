"""extend-op test report: python3 ext_report.py BATCH"""
import json, sys, statistics
V4 = "/data/fd/mingdazhang/selfedit/v4"
res = [json.loads(l) for l in open(f"{V4}/q/out/{sys.argv[1]}.jsonl")]
n = len(res); err = [r for r in res if r.get("error")]
ok = [r for r in res if not r.get("error")]
ext = [r for r in ok if r["op_trace"][0].get("extended")]
added, at_once, changed, fb, consts, flips = [], 0, 0, 0, [], [0, 0]
for r in ok:
    t = r["op_trace"][0]
    if t.get("answer_changed"):
        changed += 1
    if t.get("extended"):
        rs = t.get("rounds") or []
        added.append(sum(x.get("tokens") or 0 for x in rs))
        at_once += bool(rs and rs[0].get("closed_at_once"))
        fb += bool(t.get("fallback"))
        pt = rs and (rs[0].get("part1_prompt_tokens") or rs[0].get("prompt_tokens"))
        if pt and t.get("mode") != "cutwait":
            consts.append(pt - (r.get("prompt_tokens") or 0) - t["base_reason_tokens"] - 2)
        b, a = bool(t.get("base_ok")), bool(r.get("ok"))
        flips[0] += (not b and a); flips[1] += (b and not a)
    print(f"{r['row'][:55]:55s} base_rt {t.get('base_reason_tokens')!s:>6} ext {int(bool(t.get('extended')))} "
          f"rounds {[(x.get('part1_tokens'), x.get('reason_tokens'), x.get('tokens'), x.get('finish'), str(x.get('pred'))[:10]) for x in t.get('rounds') or []]} "
          f"base {str(t.get('base_pred'))[:10]}->{str(r.get('pred'))[:10]} ok {int(bool(t.get('base_ok')))}->{int(bool(r.get('ok')))} {t.get('fallback') or ''}")
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None
print(f"\nresults {n}, errors {len(err)}; extended {len(ext)}/{len(ok)} ({len(ext)/max(1,len(ok)):.0%}); added tokens median {statistics.median(added) if added else None} "
      f"p90 {q(added, .9)}; closing at once (<30 added reasoning tokens) {at_once}/{len(ext)}; answers changed {changed}; fallbacks {fb}; "
      f"flips wrong->right {flips[0]} right->wrong {flips[1]}; prompt-token constant (ext - base - base_reasoning - cue) {sorted(set(consts))}")
for r in err[:5]:
    print("ERR", r["row"], r["error"][:200])
