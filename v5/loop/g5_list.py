"""Generation-5 candidates for the coordinator (11-10 23:25): judged with reads missing (rows in hand / routed, d, p) and the routed op-free
candidates with d > 0 (H1 members). usage: g5_list.py RUN_DIR [G]"""
import json, os, sys
st = json.load(open(os.path.join(sys.argv[1], "state.json")))
g = sys.argv[2] if len(sys.argv) > 2 else "5"
G = st["gens"][g]
print(f"generation {g}: {len(G['cands'])} candidates; tip_change {G.get('tip_change')}")
for c in sorted(G["cands"], key=lambda c: c["cid"]):
    if not c.get("rung"):
        continue
    e = c.get("entry") or {}
    j = c.get("final") or c.get("judge") or {}
    dec = c.get("decision") or {}
    d, p, n = j.get("D", j.get("d")), j.get("p"), j.get("n_rows") or j.get("n")
    hand = c.get("rows_in_hand")
    rr = (c.get("rows_read") or {}).get("1")
    glob = e.get("cond") == "always" or (c.get("share") or 0) >= 1
    tags = ",".join(t for t, b in (("global", glob), ("ops", bool(e.get("ops"))), ("cancelled", bool(c.get("reads_cancelled"))),
                                    ("accepted", bool(c.get("accepted"))), ("final", bool(c.get("final")))) if b)
    print(f"{c['cid']} {c['author']:3s} {c.get('kind') or '':8s} routed {c.get('routed')} rows_read {rr} in_hand {len(hand) if hand is not None else '-'} "
          f"pend@dec {dec.get('pend')} d {d if d is None else round(d, 4)} p {p if p is None else round(p, 4)} n_rows {n} [{tags}] "
          f"basis: {(dec.get('basis') or '')[:60]} | {(e.get('condition') or '')[:60]}")
