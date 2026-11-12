"""Read-only extras for one generation of a v4 run (coordinator 11-10, first generation with the Stage 2 parts): lab episodes in full
(draft, what the lab returned, final answer and entry) for up to N slots of each lab author, the ops any candidate used, the weight
cells with their per-source differences, and the stack candidate's result. Never writes state.
usage: gen_extras.py RUN_DIR GEN [N_EPISODES] [ROOT]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "harness")]
import gen  # noqa: E402

RUN, g = sys.argv[1], int(sys.argv[2])
NE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
argv = ["--run", RUN] + (["--root", sys.argv[4], "--rows", os.path.join(sys.argv[4], "rows/learn.jsonl")] if len(sys.argv) > 4 else [])
R = gen.Run(gen.args(argv))
R.save = lambda: None
R.g = g
G = R.G()
C = {c.get("slot"): c for c in G["cands"] if c.get("slot") is not None and c["author"] not in ("B", "system")}
J = lambda c: c.get("final") or c.get("judge") or {}


def num(c):
    j = J(c)
    if not j.get("D") and j.get("D") != 0:
        return f"status: {c.get('status')}"
    s = (f"rows {j.get('n')} share {c.get('share', 0):.3f} d {j['D']:+.4f} se {j.get('se_used', j.get('se')):.4f} p {j.get('p')} BH {j.get('bh')} "
         f"(K {j.get('K')}, {j.get('group')}) lb {j.get('lb')} contribution_lb {j.get('contribution_lb')} | accepted {bool(c.get('accepted'))} merged {bool(c.get('merged'))}")
    return s


def per_source(c):
    src = J(c).get("src") or (c.get("judge") or {}).get("src") or {}
    return ", ".join(f"{b}: {v[0] / v[1]:+.3f} ({v[1]} rows)" for b, v in sorted(src.items()) if v[1]) or "n/a"


print(f"EXTRAS for generation {g} of {RUN} (tip at start {G['cands'][0]['parent'] if G.get('cands') else '?'})")
# 1. lab episodes
recs = R.lab_load()
by_author = {}
for p in sorted(G.get("props", []), key=lambda p: p["slot"]):
    if p.get("lab"):
        by_author.setdefault(p["author"], []).append(p)
print("\n=== LAB: slots with lab turns per author:", {a: len(v) for a, v in by_author.items()}, "| totals:", json.dumps(G.get("lab")))
for a, ps in by_author.items():
    shown = 0
    ps = sorted(ps, key=lambda p: (not any(t.get("answered") for t in p["lab"]), p["slot"]))
    for p in ps:
        if shown >= NE:
            break
        shown += 1
        c = C.get(p["slot"])
        print(f"\n--- {a} slot {p['slot']} (card source {p.get('card_source') or (c or {}).get('card_source')}), candidate {c['cid'] if c else 'none'} ---")
        x0 = R.io.get(p["key"]) or {}
        print("[DRAFT: the proposer's answer before the lab]\n" + (x0.get("content") or "(no answer)"))
        for t in p["lab"]:
            r = recs.get((t["turn"], p["slot"])) or {}
            if t.get("skip"):
                print(f"[LAB TURN {t['turn']}: skipped] {t['skip']}")
                continue
            fb = ((r.get("req") or {}).get("messages") or [{}])[-1].get("content")
            print(f"[LAB TURN {t['turn']}: what the lab returned to the proposer] (shown rows routed {t.get('shown', {}).get('routed')}, other rows routed "
                  f"{t.get('other', {}).get('routed')}, read tokens {t.get('read_tokens')})\n{fb}")
            xt = R.io.get(t["key"]) if t.get("key") else None
            print(f"[ANSWER AFTER LAB TURN {t['turn']}] ({t.get('answer')})\n" + ((xt or {}).get("content") or "(no answer)"))
        if c:
            print("[FINAL ENTRY as validated]\n" + json.dumps(c.get("entry"), ensure_ascii=False, indent=1))
            print("[RESULT] " + num(c))
# 2. ops
print("\n=== OPS used by any candidate")
n_ops = 0
for c in G["cands"]:
    e = c.get("entry") or {}
    ops = e.get("ops") or [o for x in e.get("parts") or [] for o in x.get("ops") or []]
    if ops:
        n_ops += 1
        print(f"- {c['cid']} {c['author']} kind {c.get('kind')} ops {json.dumps(ops)} | condition {json.dumps(e.get('condition'), ensure_ascii=False)[:160]}"
              f"\n    procedure {json.dumps(e.get('procedure'), ensure_ascii=False)[:300]}\n    {num(c) if c.get('rung') else 'not read: ' + str(c.get('status'))[:200]}")
props_ops = sum(1 for p in G.get("props", []) if "ops" in json.dumps((R.io.get(p.get("key_final") or p.get("key") or "") or {}).get("content") or ""))
print(f"candidates with ops: {n_ops}; proposer answers mentioning ops: {props_ops} of {len(G.get('props', []))}")
# 3. weight cells
print("\n=== WEIGHT CELLS (author B, engine W)")
for c in G["cands"]:
    if c.get("target") == "E.W" or (c.get("entry") or {}).get("edit"):
        ec = c.get("ecomp") or {}
        print(f"- {c['cid']} cell {c.get('cell')} ({c.get('cell_group')}) edit {json.dumps((c.get('entry') or {}).get('edit'))}\n    {num(c)}"
              f"\n    per source d: {per_source(c)}\n    LOBO {J(c).get('lobo')} | tok {c.get('tok_cand') or 0:.0f} vs parent {c.get('tok_parent') or 0:.0f} | "
              f"error rows {ec.get('errors')} op_skipped {ec.get('skipped')} | status {c.get('status')}")
# 4. stack
print("\n=== STACK (author system)")
for c in G["cands"]:
    if c["author"] == "system":
        print(f"- {c['cid']} parts {c.get('stack_of')} routed {c.get('routed')} share {c.get('share')}\n    {num(c)}\n    per source d: {per_source(c)}"
              f"\n    tok {c.get('tok_cand') or 0:.0f} vs parent {c.get('tok_parent') or 0:.0f} | status {c.get('status')}")
        for x in (c.get("entry") or {}).get("parts") or []:
            print(f"    part {x.get('cid')}: cond {json.dumps(x.get('condition'), ensure_ascii=False)[:160]} settings {json.dumps(x.get('settings'))} ops {x.get('ops')}")
