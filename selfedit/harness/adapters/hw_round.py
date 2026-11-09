# selfedit 1109c: NEW adapter (§3.1, §8.2 items 2, 5, 6, 7, 8, 9): the H/W round glue that hboost.Run.round calls under prereg block "c1109".
# Every function takes the Run (R) and the hboost module (H); nothing here edits hboost state except through the hooks it is called from.
from __future__ import annotations

import collections
import json
import math
import os
import re
import time
import zlib

import numpy as np

import harness
from harness.anchor import eproc
from harness.anchor.config import RRSIConfig
from harness.anchor.history import History, exploration, stall_flag
from harness.anchor.schedule import edit_budget
from harness.anchor.stats import holm_bonferroni
from harness.adapters import slot_bandit as SB
from harness.adapters import member_cand as MC, pool_score as PS
from harness.adapters.meta_level import Anchor, MetaLevel, valid_policy
from harness.adapters.qwen_llm import QwenLLM
from harness.adapters.skill_store import SkillStore, precheck
from harness.ops import admit as ADMIT

HDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = dict(on=True, mech_cells=True, ledger=True, rung="J1", null_draws=(4, 6), stage2_cap=6, fired_min=8, skills=True, reflect=True, meta_p=True, critic=True, rotate=True, rotate_seed=1109, eproc=True, prescreen=True, null_ref=True, a5=False, null_pairs=((4, 8), (6, 9)), a6=False, a6b=True, a6p=False, a7=False, harness_ab=False, a8=False, carry_cap=4, carry_seed=(), a9=False, knob_ops=("tokens", "verbosity"), knob_min=2, allocator="p_level", gate_key="cat", gate_key_min=40, gate_prompts="", gate_control_note="", gate_recite_min=.95, gate_misfire_max=.01, s1_net=False, early_a7=False)
PRE = json.load(open(os.path.join(HDIR, "frozen", "prereg_1109c.json")))
MENU_BASE = ("tokens",)      # fable_1109d patch 4: the knobs (scale, skip, attn_temp, vision_scale, direction, verbosity, tune) leave the menu; patch (+ kernel) and gate enter with their admission


def cfg(R):
    c = dict(FIELDS, **((R.a or {}).get("c1109") or {}))
    bad = sorted(set(c) - set(FIELDS) - {"note"})
    if bad:
        raise SystemExit(f"prereg c1109: unknown fields {bad}; known {sorted(FIELDS)}")
    if c["rung"] != PRE["rung"]:
        raise SystemExit(f"prereg c1109 rung {c['rung']} != frozen rung {PRE['rung']} (harness/frozen/prereg_1109c.json)")
    return c


def on(R):
    return bool((R.a or {}).get("c1109"))


def init(R, H):
    """Once per Run: anchor guard, operator registry copy in the work dir, skill store, MetaLevel, History, QwenLLM."""
    c = cfg(R)
    if R.fusion not in H.DS_MODES:
        raise SystemExit("prereg c1109 reads the fused ds pool: set fusion ds_agree")
    if (R.a or {}).get("c128", {}).get("judge"):
        raise SystemExit("prereg c1109 replaces the c128 judge: do not set c128 judge")
    ev = os.path.join(R.work, "evolvable")
    os.makedirs(ev, exist_ok=True)
    R.c9 = c
    R.c9_anchor = Anchor(HDIR, R.state)
    R.c9_anchor.check("start")
    R.c9_skills = SkillStore(os.path.join(ev, "skills"))
    R.c9_hist = History(os.path.join(ev, "history.jsonl"))
    R.c9_notes = os.path.join(ev, "proposer_notes.md")
    reg = os.path.join(ev, "registry.json")
    if not os.path.exists(reg):
        open(reg, "w").write(open(os.path.join(HDIR, "ops", "registry.json")).read())
    R.c9_reg = reg
    R.c9_meta = MetaLevel(R.state, R.a["slots"], lambda: R.state.get("c1109_delta") or 0.0)
    R.c9_cfg = RRSIConfig(**PRE["rrsi_cfg"])
    H.MENU_OPS[0] = list(MENU_BASE)          # fable_1109d patch 4: patch, tokens, program (+ admitted W families at each round start)
    MC.NULLREF[0] = bool(c.get("null_ref", True))     # amendment 4 (11-09)
    MC.S4[0] = bool(c.get("a5"))                      # amendment 5 (11-09): member statistics on 4 samples (own read + seed-paired read)
    A5_GLOBAL[0] = bool(c.get("a5"))                  # amendment 5: e-process for global members (provisional line without the scoped-only clause)
    MC.A6P[0] = bool(c.get("a6p"))                    # amendment 6': only a gain-path null voids the round
    MC.A6B[0] = bool(c.get("a6b", True))              # amendment 6 (b): dominance void (off: (a) only)
    MC.A6[0] = bool(c.get("a6"))                      # amendment 6 (11-09): null rule = candidate rule; a null voids only the admits that do not beat it
    if c.get("mech_cells", True):            # A5 (fable_1109d patch 13): mechanism fusion cells with back-off
        R.fuser.MECH[0] = True
    R.c9_ledger = {}
    R.c9_bench_out = None


def llm(R, rd):
    return QwenLLM(R, os.path.join(rd, "c1109")).bind()


# ------------------------------------------------------------------ round start: rotation, admission flags, plan --------------------------------
def round_start(R, H, r):
    R.c9_anchor.check(f"round {r} proposals")
    reg = ADMIT.admit(R.c9_reg)
    R.c9_wops = ADMIT.admitted_ops(reg)
    H.MENU_OPS[0] = list(MENU_BASE) + [o for o in R.c9_wops if o not in MENU_BASE] + (["kernel"] if "patch" in R.c9_wops else [])
    if R.c9.get("a9") or R.c9.get("allocator") == "ts":       # amendment 9 / 10: the grid knobs tokens + verbosity are proposer operators with their own slots
        H.MENU_OPS[0] += [o for o in R.c9["knob_ops"] if o not in H.MENU_OPS[0]]
    H.jdump(f"{R.work}/r{r:03d}/c1109_registry.json", reg)
    R.say(f"r{r}: 1109c admitted W families: {R.c9_wops or 'none'} ({ {n: f['why'] for n, f in reg['families'].items()} })")
    if R.c9["rotate"]:
        out = R.data.rotate(r, int(R.c9["rotate_seed"]))
        H.jdump(f"{R.work}/r{r:03d}/rotation.json", dict(round=r, benches=out, select=R.data.of("select")))
        R.say(f"r{r}: 1109c SELECT rotation (seed {R.c9['rotate_seed']}): {out}")


def plan(R, r, rng, can_cross, have_grid=True):
    """Slot operators. c1109 allocator: "p_level" (default: the MetaLevel P split, below) or "ts" (amendment 10: slot_bandit, plan_ts)."""
    if R.c9.get("allocator") == "ts":
        return plan_ts(R, r, can_cross)
    return plan_p(R, r, rng, can_cross, have_grid)


def plan_ts(R, r, can_cross):
    """Amendment 10 (fable_1109_slot_allocator.md §2): Thompson sampling over operator families. Prior from the round-1 stock grid (debiased best
    cell), ledger of per-proposal rewards (round_end, A8 revisions), one floor slot per available family, the rest by posterior draws (cap 6),
    then a permutation. Its own rng (seed rotate_seed, round), so the proposer's rng stream is unchanged."""
    S = R.state.setdefault("slot_bandit", dict(ledger={}, prior=None, deals=[]))
    if S["prior"] is None:
        g = H_grid(R)
        S["prior"] = SB.grid_prior(g.get("rows_of") if g else [])
    avail = ["program"] + (["cross"] if can_cross else []) + [o for o in ("patch", "gate") if o in (R.c9_wops or [])] + list(R.c9["knob_ops"])
    post = SB.posterior(S["ledger"], S["prior"], r)
    rng_b = np.random.default_rng([int(R.c9["rotate_seed"]), r, 10])
    out, rows = SB.deal(post, avail, R.a["slots"], rng_b)
    pb = SB.p_best(post, avail, np.random.default_rng([int(R.c9["rotate_seed"]), r, 11]), n=1000)
    for k, x in enumerate(rows):
        x.update(round=r, slot=k, avail=avail, post=post[x["family"]])
    S["deals"].append(dict(round=r, deal=dict(collections.Counter(out)), post=post, p_best=pb, rows=rows))
    side = {k: rows[k]["mode"] for k in range(len(rows))}
    R.c9_side = side
    R.state.setdefault("c1109_sides", {})[str(r)] = side
    R.say(f"r{r}: 1109c TS slots {dict(collections.Counter(out))} (floor {sum(1 for x in rows if x['mode'] == 'floor')}, ts {sum(1 for x in rows if x['mode'] == 'ts')}; avail {avail}; "
          f"posterior mean " + ", ".join(f"{k} {post[k]['m']:+.2f} (kappa {post[k]['kappa']:.1f}, valid {post[k]['V']:.1f}/{post[k]['V'] + post[k]['I']:.1f})" for k in avail) + f"; P(best) {pb})")
    return out


def H_grid(R):
    f = f"{R.work}/r001/grid.json"
    return json.load(open(f)) if os.path.exists(f) else None


def bandit_update(R, r, reqs, props, rec):
    """Amendment 10: the round's per-proposal rewards into the ledger (stage 2: member-track D x 100 on its rows; stage 1 only: 4 x zx, w .13;
    invalid: validity failure). Cost-adjusted by the RRSI rate (10 pp per unit dC above .1)."""
    if R.c9.get("allocator") != "ts":
        return []
    S = R.state.setdefault("slot_bandit", dict(ledger={}, prior=None, deals=[]))
    by_slot = {q["slot"]: q for q in reqs if q.get("id") != "meta"}
    out = []
    for p in props:
        q = by_slot.get(p.get("slot"), {})
        fam = SB.family_of(q.get("op") or p.get("op"))
        if fam is None or not p.get("name"):
            continue
        x = p.get("c1109") or None
        a7 = (x or {}).get("a7") or {}
        dC = ((x["C1"] - x["C0"]) / x["C0"]) if x and x.get("C0") and x.get("C1") else 0.0
        if not p.get("valid"):
            SB.record(S["ledger"], p["name"], fam, r, False, why=str(p.get("status"))[:120])
        elif a7.get("path") == "member" and a7.get("rows"):
            SB.record(S["ledger"], p["name"], fam, r, True, y=100 * a7["D"], w=min(2.0, a7["rows"] * 4 / 1600), dC=dC, stage=2)
        elif p.get("zx_score") is not None:
            SB.record(S["ledger"], p["name"], fam, r, True, y=4 * float(p["zx_score"]), w=SB.W_STAGE1, dC=dC, stage=1)
        else:
            continue
        out.append(dict(name=p["name"], **S["ledger"][p["name"]]))
    return out


def bandit_revise(R, r, name, D_acc, rows_acc, dC=0.0):
    """Amendment 10 x A8: a carried candidate's record is revised in place from its accumulated fresh reads."""
    S = R.state.get("slot_bandit")
    if not S or name not in S["ledger"]:
        return
    x = S["ledger"][name]
    SB.record(S["ledger"], name, x["family"], x["round"], True, y=100 * D_acc, w=min(2.0, rows_acc * 4 / 1600), dC=dC, stage=f"carry")
    S["ledger"][name]["revised_round"] = r


def plan_p(R, r, rng, can_cross, have_grid=True):
    """Slot operators from the P level: per slot the side (incumbent / challenger) deals program, cross and W slots; W slots go round-robin over the
    admitted families; with none admitted (or no grid this round) they become program slots (logged)."""
    side = R.c9_meta.plans(r)
    wops = R.c9_wops if have_grid else []
    lists = {}
    for s in ("inc", "ch"):
        p = R.c9_meta.policy(s)
        if not p:
            continue
        cro = p["cross"] if can_cross else 0
        L = ["program"] * (p["program"] + p["cross"] - cro) + ["cross"] * cro
        w = [wops[k % len(wops)] for k in range(p["w"])] if wops else []
        L += w + ["program"] * (p["w"] - len(w))
        lists[s] = [L[i] for i in rng.permutation(len(L))]
    out = [lists[side[k]][k] for k in range(R.a["slots"])]
    if R.c9.get("a9"):                       # amendment 9 (11-09): knob_min slots per round for the grid knobs (tokens, verbosity, in turn), taken from program slots
        knobs = list(R.c9["knob_ops"]); prog = [k for k, o in enumerate(out) if o == "program"]
        for j, k in enumerate(prog[-int(R.c9["knob_min"]):] if prog else []):
            out[k] = knobs[j % len(knobs)]
    R.c9_side = side
    R.state.setdefault("c1109_sides", {})[str(r)] = side
    R.say(f"r{r}: 1109c slots {dict(collections.Counter(out))} (sides {dict(collections.Counter(side.values()))}; W families {wops or 'none'}; cross possible {can_cross})")
    return out


AB_WINDOWS = {3: "notes", 4: "notes", 5: "skills", 6: "skills", 7: "p_challenger", 8: "p_challenger", 9: "ledger", 10: "ledger"}   # amendment 7 / fable_1109 §3


def ab_window(R, r):
    """The A/B component of round r. Under the TS allocator (amendment 10) the P-challenger window r7-8 repeats the more ambiguous of notes / skills
    (one-sided Wilcoxon p closest to .10), since the allocator replaces the P A/B."""
    w = AB_WINDOWS.get(r)
    if w == "p_challenger" and R.c9.get("allocator") == "ts":
        dec = R.state.get("c1109_ab_dec", {})
        c = [k for k in ("notes", "skills") if k in dec]
        return min(c, key=lambda k: abs(float(dec[k]["p"]) - .10)) if c else "notes"
    return w


def ab_side(R, r, k):
    """Amendment 7 slot-parity A/B: inside a component's 2-round window the component is on for odd slots only; from r11 a component is on iff it won
    its window (one-sided Wilcoxon rank-sum on q, alpha .10); before r3 all on. Off when c1109 harness_ab is false (all on). Logged per (round, slot)."""
    on = dict(notes=True, skills=True, ledger=True)
    win = ab_window(R, r) if R.c9.get("harness_ab") else None
    if win in on:
        on[win] = bool(k % 2 == 1)
    elif R.c9.get("harness_ab") and r > max(AB_WINDOWS):
        dec = R.state.get("c1109_ab_dec", {})
        for c in on:
            on[c] = bool(dec.get(c, {}).get("on", True))
    R.state.setdefault("c1109_ab", {}).setdefault(str(r), {})[str(k)] = dict(window=win, **on)
    return on


def prompt_extra(R, H, r, k, op):
    """Text appended to a PROPOSER prompt: proposer notes (Self-HPO) and <= 2 matched skills (code chooses); RRSI prune set / exploration text."""
    ex = ""
    on = ab_side(R, r, k)
    if on["notes"] and os.path.exists(R.c9_notes):
        ex += "\n\n## Your notes from earlier rounds (proposer_notes.md, written by you after each round's verdicts)\n" + open(R.c9_notes).read()[-3000:]
    t = r - 1
    traj = R.state.get("c1109_traj", [])
    sig = stall_flag(traj, len(traj) - 1, 3, R.state.get("c1109_delta") or 0.0) if traj else 0
    E = exploration(t, sig, R.c9_hist.tried(), 1)
    B = R.c9_hist.prune_set(t, R.c9_cfg.n_prune)
    ex += "\n\n## Search state (code-written)\n" + E["text"] + ("\nComponents with no recent gain (prune set): " + ", ".join(b["component"] for b in B) if B else "")
    ex += f"\nEdit budget this round: at most {edit_budget(t, R.a.get('rounds', 20), R.c9_cfg.b_min, R.c9_cfg.b_max)} engine edits in one proposal."
    ex += ("\nRejected before any engine time (leakage precheck, 11-09): a program or note that states a fixed value for a kind of object or scene "
           "(a number next to legs, pieces, rings, sides, squares, columns, wheels or players; 'standard/usually/typically ... <number>'; 'name = <number>'), "
           "an option letter given as an answer, a bench name in a rule. Describe the procedure (enumerate, read, compare) without the value.")
    if R.c9.get("ledger", True) and on["ledger"]:           # fable_1109d patch 7: machine-made ledger of the pool's own traces on SHOW rows (built once per round)
        if r not in R.c9_ledger:
            from harness.hw import experience as EXP
            mem = R.state["members"]
            try:
                R.c9_ledger[r] = EXP.build_ledger(R.data, R.votes(mem), R.data.of("show"))["text"]
            except Exception as e:          # a failed ledger leaves the prompt without it, logged
                R.c9_ledger[r] = ""
                R.say(f"r{r}: 1109c ledger failed, prompts go without it: {type(e).__name__}: {str(e)[:200]}")
        ex += "\n\n" + R.c9_ledger[r] if R.c9_ledger[r] else ""
    sk = R.c9_skills.match(force_unused=bool(sig) and k % 6 == 5) if R.c9["skills"] and on["skills"] else []
    ex += SkillStore.block(sk)
    return ex, [s["front"]["id"] for s in sk]


def check_cards(R, req):
    bad = [i for i in (req.get("card_rows") or []) if R.data.role.get(i) != "show"]
    if bad:
        raise AssertionError(f"1109c: card rows must be show rows only, slot {req.get('slot')} has {len(bad)} others, e.g. {bad[:3]}")


# ------------------------------------------------------------------ critic (before engine time) ------------------------------------------------
def critic(R, H, r, rd, kids):
    """Regex precheck (denylist + hboost hardcoded_values) then the frozen-model LEAK critic (hboost CRITIC_*), one batch. A hit or 'LEAK: yes' drops the proposal
    with the reason; a failed call or missing verdict is retried once, then the proposal is dropped as unchecked (fail closed)."""
    if not R.c9["critic"]:
        return kids
    live = []
    for m, row in kids:
        if not row.get("valid"):
            continue
        txt = H.critic_text(m)
        hits = precheck((m.get("gate") or {}).get("note") or "") + [f"hard-coded value {h}" for h in H.hardcoded_values(m.get("program") or "")[:3]]
        if hits:
            row.update(valid=False, status="invalid: leakage precheck: " + "; ".join(hits), leak_precheck=hits)
            continue
        live.append((m, row, txt))
    if live:                 # 1109c-final: fail closed. One retry (own phase) for a failed call or a missing verdict; still unchecked -> invalid, never passed
        todo, L = list(live), llm(R, rd)
        for k_, ph in enumerate(("critic", "critic_retry")):
            if not todo:
                break
            try:
                got = L.generate_many(ph, [dict(id=f"leak{row['slot']}", system=H.CRITIC_SYSTEM, user=H.CRITIC_ASK + "\n\n" + txt) for m, row, txt in todo])
            except (RuntimeError, OSError, H.Bad) as e:
                R.say(f"r{r}: 1109c critic call failed ({ph}): {str(e)[:200]}")
                got = {}
            left = []
            for m, row, txt in todo:
                ans = (got.get(f"leak{row['slot']}") or {}).get("content")
                v = H.critic_verdict(ans)
                row["leak_critic"] = v
                if v == "yes":
                    row.update(valid=False, status="invalid: leakage critic: " + str(ans)[:300])
                elif v != "no":
                    left.append((m, row, txt))
            todo = left
        for m, row, _ in todo:
            row.update(valid=False, leak_critic="unchecked", status="invalid: leakage critic unchecked (call failed or no verdict twice): fail closed")
        if todo:
            R.say(f"r{r}: 1109c critic unchecked after retry, dropped (fail closed): {[row.get('name') for _, row, _ in todo]}")
    dropped = [(row.get("name"), str(row.get("status"))[:80]) for m, row in kids if not row.get("valid") and "leakage" in str(row.get("status"))]
    R.say(f"r{r}: 1109c critic dropped {len(dropped)}: {dropped}")
    return [(m, row) for m, row in kids if row.get("valid")]


# ------------------------------------------------------------------ null draws -----------------------------------------------------------------
def null_members(R, H, m):
    ep = int((R.state.get("c1109_epoch") or {}).get(m["name"], 0))
    return [H.draw_of(m, d + H.DRAW_EPOCH * ep) for d in R.c9["null_draws"]]


def pair_of(R, H, x, parent=None):
    """Amendment 5: the seed-paired second read of x. A null draw d of a parent -> the parent's draw null_pairs[d] (same epoch block); a member or a
    candidate -> its draw PAIR_DRAW. None when a5 is off."""
    if not R.c9.get("a5"):
        return None
    d = (x.get("draw") or {}).get("d")
    if parent is not None and d is not None:
        ep, d0 = divmod(int(d), H.DRAW_EPOCH)
        pr = dict((int(a), int(b)) for a, b in R.c9["null_pairs"])
        return H.draw_of(parent, pr[d0] + H.DRAW_EPOCH * ep) if d0 in pr else None
    return H.draw_of(x, PAIR_DRAW)


def four_map(R, H, members, cands=()):
    """Amendment 5: name -> paired read (member dict) for every member, every member's null draws and every candidate; MC.FOUR set from it."""
    out = {}
    for m in members:
        out[m["name"]] = pair_of(R, H, m)
        for x in null_members(R, H, m):
            out[x["name"]] = pair_of(R, H, x, m)
    for m in cands:
        out[m["name"]] = pair_of(R, H, m)
    out = {k: v for k, v in out.items() if v is not None}
    MC.FOUR.clear(); MC.FOUR.update({k: v["name"] for k, v in out.items()})
    return out


def control_of(m):
    """gate_v3 (1109c-final): the gate write's matched wrong-content control as a member (same parent, program, scope and inherited gates;
    engine = parent + the control arm from the same mm_gate job). None when m has no control; {"missing": why} stays None too."""
    c = m.get("gate_control") or {}
    if not c.get("mid"):
        return None
    return dict({k: v for k, v in m.items() if k not in ("gate_control", "gate_control_file")}, name=c["name"], mid=c["mid"], genome=c["genome"],
                gate=dict(m.get("gate") or {}, note=c.get("note", "")), note=c.get("note", ""), origin="gate_control")


def controls(top):
    return [c for c in (control_of(m) for m, _ in top) if c is not None]


def own_scope(R, H, m, i):
    """Row i is in m's OWN gate scope (not an inherited one)."""
    s = (m.get("gate") or {}).get("scope") or ""
    return s.startswith("cat:") and s == f"cat:{H.vh.key(R.data.rows[i])}"


def gold_above_chance(R, H, v4, ids):
    """gate_v3: per gold value on ids, the candidate's mean right rate > chance (1/len(options), else 1/number of distinct gold values on ids).
    Returns (ok, {gold: [rows, acc, chance]})."""
    rows = [i for i in ids if H.right_rate(v4.get(i)) is not None]
    gold = {i: str(R.data.rows[i].get("answer")) for i in rows}
    ng = max(1, len(set(gold.values())))
    out = {}
    for g in sorted(set(gold.values())):
        I = [i for i in rows if gold[i] == g]
        opts = [len(R.data.rows[i].get("options") or []) for i in I]
        ch = float(np.mean([1.0 / o if o else 1.0 / ng for o in opts]))
        out[g] = [len(I), float(np.mean([H.right_rate(v4[i]) for i in I])), ch]
    return bool(out) and all(a > c for _, a, c in out.values()), out


def request_nulls(R, H, members, top=()):
    """Each member's real re-seeded draws on score + SELECT (cached on disk: only missing rows run). Amendment 5: also every paired read (members,
    their nulls, the stage-2 candidates in top)."""
    rows = sorted(set(R.data.of("score")) | set(R.data.of("select")))
    R.no_id(rows, "null draws")
    want = [(x, rows) for m in members for x in null_members(R, H, m)]
    want += [(x, rows) for x in four_map(R, H, members, [m for m, _ in top]).values()]
    want += [(x, rows) for x in carry_reads(R, H, top)]          # amendment 8
    for c in controls(top):          # gate_v3: the control's base samples and its seed-paired draw (four_map registers the pair)
        want += [(c, rows)] + ([(pair_of(R, H, c), rows)] if pair_of(R, H, c) is not None else [])
    R.ev.evaluate_many(want)


# ------------------------------------------------------------------ stage 1 (A1: fire gate + zx ranking) ----------------------------------------
def screen_ok(R, fired, fmin, net, zx):
    """Stage-1 screen: fired >= fmin and zx > 0 (1109c audit 11-09); amendment 11 (coordinator 11-10 ~02:05 CST, screen only): also net > 0, with
    fired_min 3 in the prereg. Stage 2 and every admission threshold unchanged."""
    return bool(fired >= fmin and zx is not None and float(zx) > 0 and (net > 0 or not R.c9.get("s1_net")))


def early_draw(R, H, r, members, m, row, qm):
    """Amendment 11 (coordinator 11-10): as soon as a candidate's own stage-1 rows have landed and its own screen passes (fired / net against the
    members' pool, zx vs its parent on its stage-1 rows), its A7 paired draw (pair_of, draw PAIR_DRAW) on sorted(score | select) is asked now through
    the same evaluate_many as request_nulls: same rows, chunking, out file and line seed, so the later request finds this line (.job mark) and waits on
    it -- the answers are this line's, asked once. The line is named _se (LOW) and stage 1 does not wait on it. Selection is unchanged."""
    if m.get("_early") or not R.c9.get("a5"):
        return
    m["_early"] = True
    pr = pair_of(R, H, m)
    if pr is None:
        return
    votes = R.votes(members + [m])
    fu = R.fz("fuse", R.fuser.fuse, members, votes, R.data, R.fusion)
    pa, tab = H.pool_cols(fu), fu.tab
    Q = R.no_id(list(qm(m)), "early A7")
    F = H.fired_rows(tab, votes[m["name"]], pa, Q)
    fmin = min(int(R.c9["fired_min"]), max(1, math.ceil(len(Q) / 2)))
    net = float(sum(H.vs_pool(tab, votes[m["name"]], pa, F).values()))
    zx = H.zx_stats(tab, votes[m["parents"][0]], votes[m["name"]], qm(m), R.data.cluster)["zx"]
    ok = screen_ok(R, len(F), fmin, net, zx)
    row["c1109_early"] = dict(fired=len(F), net=net, zx=zx, asked=ok)
    if not ok:
        return
    rows = R.no_id(sorted(set(R.data.of("score")) | set(R.data.of("select"))), "early A7")
    n0, s0 = len(R.ev.pending), set(R.ev.sent)
    R.ev.early = True
    try:
        R.ev.evaluate_many([(pr, rows)])
    finally:
        R.ev.early = False
    new = R.ev.pending[n0:]
    del R.ev.pending[n0:]
    for o in set(R.ev.sent) - s0:
        R.ev.sent.pop(o, None)
    R.say(f"  1109c A11 early A7 draw {pr['name']}: own screen fired {len(F)} net {net:+.0f} zx {zx if zx is None else round(zx, 2)} -> asked as LOW {sorted({j['name'] for j in new})}")


def stage1(R, H, r, rd, members, have, qm):
    votes = R.votes(members + [m for m, _ in have])
    fu = R.fz("fuse", R.fuser.fuse, members, votes, R.data, R.fusion)
    pa, tab = H.pool_cols(fu), fu.tab
    rows = []
    for m, row in have:
        Q = R.no_id(list(qm(m)), "stage 1")
        F = H.fired_rows(tab, votes[m["name"]], pa, Q)
        fmin = min(int(R.c9["fired_min"]), max(1, math.ceil(len(Q) / 2)))       # 1109c audit 11-09: a narrow (gate) member's fire gate scales to its own scope rows
        zx = row.get("zx_score")
        net = float(sum(H.vs_pool(tab, votes[m["name"]], pa, F).values()))
        row["c1109_s1"] = dict(fired=len(F), scope_rows=len(Q), fired_min=fmin, net=net, zx=zx, passes=screen_ok(R, len(F), fmin, net, zx))
        rows.append(dict(name=m["name"], **row["c1109_s1"]))
    ok = sorted([x for x in have if x[1]["c1109_s1"]["passes"]], key=lambda x: -float(x[1].get("zx_score") or 0))
    top = ok[: int(R.c9["stage2_cap"])]
    for _, row in ok[int(R.c9["stage2_cap"]):]:
        row["stage2_by"] = "over the stage-2 cap"
    for _, row in top:
        row["stage2_by"] = "1109c fire gate + zx"
    H.jdump(f"{rd}/c1109_stage1.json", rows)
    R.say(f"r{r}: 1109c stage 1 (fire gate >= {R.c9['fired_min']}, zx ranks, cap {R.c9['stage2_cap']}): stage 2 {[m['name'] for m, _ in top]}")
    return top


# ------------------------------------------------------------------ stage 2 decide ---------------------------------------------------------------
def lineage_cards(R, m, extra=()):
    """Card rows shown to the proposers of m and of each ancestor (hist rows carry card_rows; members carry parents)."""
    hist = {x.get("name"): x for x in R.state.get("hist", []) if x.get("name")}
    by = {x["name"]: x for x in list(R.state.get("members", [])) + list(extra)}
    out, seen, x = set(), set(), m
    while x is not None and x["name"] not in seen:
        seen.add(x["name"])
        out |= {i for i in (hist.get(x["name"], {}).get("card_rows") or x.get("card_rows") or []) if i in R.data.role}
        p = (x.get("parents") or [None])[0]
        x = by.get(p) if p else None
    return out


DELTA_MIN_ROWS = 150      # A2 (fable_1109d §1.3 item 2; astra 1109d): delta from >= 150 covered rows, else the last such delta, else the bootstrap sd of the base read


DELTA_MIN_EVALS = 6


def delta_now(R, H, members, votes, Rb):
    from harness.anchor import calibrate as rcal
    from harness.anchor.evaluate import TaskResult, aggregate
    NL = {m["name"]: null_members(R, H, m) for m in members}            # as cal_1109c delta_of: redraw j of each member where it exists, else the member
    J = max([len(v) for v in NL.values()] + [0])
    sets = [list(members)] + [[NL[m["name"]][j] if len(NL[m["name"]]) > j else m for m in members] for j in range(J)]
    cov = [i for i in Rb if all(votes[x["name"]].get(i) for s in sets for x in s)]
    def ev(s, rows, k):
        ids, right = PS.pool_S(H, R, s, votes, rows)
        return aggregate(f"inc{k}", 1, {i: TaskResult(rewards=[float(v)]) for i, v in zip(ids, right)})
    if len(cov) >= DELTA_MIN_ROWS and len(sets) >= 2:
        cal = rcal.calibrate([ev(s, cov, k) for k, s in enumerate(sets)], z=R.c9_cfg.delta_z, reps=500)
        if len(sets) < DELTA_MIN_EVALS:     # 1109c audit 11-09: the sd of < 6 repeated reads (chi2 with <= 4 df) is too wide and too noisy; the bootstrap sd of the pooled reads is used, the redraw sd logged
            cal = dict(cal, delta_redraw=cal["delta"], delta=round(R.c9_cfg.delta_z * cal["sd_null_bootstrap"], 6), sd_use="bootstrap (fewer than %d reads)" % DELTA_MIN_EVALS)
        R.state["c1109_delta_last"] = float(cal["delta"])
        return float(cal["delta"]), dict(cal, rows=len(cov), delta_rows=len(cov), source="redraws")
    last = R.state.get("c1109_delta_last")
    if last is not None:
        return float(last), dict(rows=len(cov), delta_rows=len(cov), source=f"last delta on >= {DELTA_MIN_ROWS} rows (this round {len(cov)} covered)")
    cal = rcal.calibrate([ev(list(members), [i for i in Rb if all(votes[m["name"]].get(i) for m in members)], 0)], z=R.c9_cfg.delta_z, reps=500)
    return float(cal["delta"]), dict(cal, rows=len(cov), delta_rows=len(cov), source="bootstrap sd of the base read (no >= 150-row redraw cover yet)")


def _rebuild(R, H, r, n, hist, by, what):
    """A stage-2 candidate of an earlier round rebuilt from its own stored proposal (same id), with its row; None (logged) if it cannot be."""
    h = hist.get(n)
    if h is None:
        R.say(f"r{r}: 1109c {what} {n}: no record, skipped"); return None
    r0, rd0 = int(h["round"]), f"{R.work}/r{int(h['round']):03d}"
    q = {x["slot"]: x for x in json.load(open(f"{rd0}/prompts.json")) if x.get("id") != "meta"}[h["slot"]]
    f = next((f"{rd0}/answers/{fid}.json" for fid in (q["id"] + "_re", q["id"]) if os.path.exists(f"{rd0}/answers/{fid}.json")), None)
    par = [by[p] for p in by if p in (h.get("parents") or ["stock"])] or [by["stock"]]
    try:
        m = R.candidate(r0, q, H.parse_answer(json.load(open(f)).get("content")), par)
    except Exception as e:
        R.say(f"r{r}: 1109c {what} {n}: rebuild failed {type(e).__name__}: {str(e)[:160]}"); return None
    m.pop("own_ops", None)
    if f"r{r0}s{h['slot']}_{m['mid'][:6]}" != n:
        R.say(f"r{r}: 1109c {what} {n}: rebuilt id {m['mid'][:6]} differs, skipped"); return None
    m["name"] = n
    return m, dict(name=n, round=r0, slot=h["slot"], card_rows=h.get("card_rows"), stage2_status="ok", zx_score=h.get("zx_score"))


# ------------------------------------------------------------------ amendment 8: cross-round e-process for carried candidates -----------------
CARRY_D0 = 1000            # carried re-read r uses draws CARRY_D0 + 2r and CARRY_D0 + 2r + 1 of candidate AND parent (same seeds: paired; fresh every round)
CARRY_MAX_READS = 4
CARRY_ALPHA = 0.05


def carry_draws(r):
    return (CARRY_D0 + 2 * r, CARRY_D0 + 2 * r + 1)


def carry_in(R, H, r, members, by):
    """Amendment 8: every carried candidate (state c1109_carry; the prereg carry_seed names join once) re-enters stage 2 this round, row carried=True;
    its fresh 4-vs-4 read is asked by request_nulls. A carried candidate whose parent left the pool is dropped (logged)."""
    if not R.c9.get("a8"):
        return []
    C = R.state.setdefault("c1109_carry", {})
    if not R.state.get("c1109_carry_seeded"):
        for n in R.c9.get("carry_seed") or ():
            C.setdefault(n, dict(since=r, reads=[], Ds=[], rows=0, harm=0.0, harm_thr=0.0, sumd=0.0, sumd2=0.0, status="carried", seeded=True))
        R.state["c1109_carry_seeded"] = True
    hist = {x["name"]: x for x in R.state.get("hist", []) if x.get("name")}
    out = []
    for n, c in list(C.items()):
        if c.get("status") != "carried":
            continue
        x = (dict(c["member"]), dict(c["row"])) if c.get("member") else _rebuild(R, H, r, n, hist, by, "carry")
        if x is None or x[0]["parents"][0] not in by:
            c["status"] = "dropped (rebuild failed or parent gone)"; continue
        m, row = x
        row.update(carried=True, carry_round=r, carry_parent=by[m["parents"][0]], stage2_status="ok", stage2_by="1109c A8 carried")
        out.append((m, row))
    if out:
        rows = sorted(set(R.data.of("score")) | set(R.data.of("select")))
        R.no_id(rows, "carry")
        R.ev.evaluate_many([(m, rows) for m, _ in out])
        R.say(f"r{r}: 1109c A8 carried into stage 2: {[m['name'] for m, _ in out]}")
    return out


def carry_reads(R, H, top):
    """The fresh paired draws of every carried candidate and its parent (request_nulls asks them with the null draws)."""
    out = []
    for m, row in top:
        if row.get("carried"):
            for d in carry_draws(int(row["carry_round"])):
                out += [H.draw_of(m, d), H.draw_of(row["carry_parent"], d)]
    return out


def carry_judge(R, H, r, m, row, rc, nulls_p, Rm, votes, four):
    """Amendment 8 read of one carried candidate: fresh 4 vs 4 (draws carry_draws(r) of candidate and parent), D per row of R (scope rows if scoped),
    one bet per picture cluster (cluster mean D), E = eproc.eprocess_rows over all fresh reads (the read that carried it is not a bet).
    Admit iff E >= K/alpha (e-Bonferroni over the fixed carry cap K) and the fixed guards on the accumulated rows: no-harm (sum of harm <= sum of the
    parent-redraw bounds), mean D > 0 (unseen rows), rows up > down, critic (passed at proposal), fired-z >= 1.96 if scoped, RRSI cost rule (gain
    branch, dS = accumulated mean D). Drop at D <= 0 on two consecutive reads or after CARRY_MAX_READS reads."""
    from harness.anchor.selection import cost_rule
    K = int(R.c9.get("carry_cap", 4))
    C = R.state["c1109_carry"][m["name"]]
    p = row["carry_parent"]
    dA, dB = carry_draws(r)
    cA, cB, pA, pB = H.draw_of(m, dA), H.draw_of(m, dB), H.draw_of(p, dA), H.draw_of(p, dB)
    v = R.votes([cA, cB, pA, pB])
    vc, vp = H.merge4(v[cA["name"]], v[cB["name"]]), H.merge4(v[pA["name"]], v[pB["name"]])
    rows = Rm if rc["global_"] else [i for i in Rm if vc.get(i)]
    mt = MC.member_track(H, R.data, vc, vp, rows, seed=r)
    if not mt["rows"]:
        C["status"] = "dropped (no rows answered)"
        return False, dict(path="carried", rows=0)
    by = {}
    for c_, x in zip(mt["clusters"], mt["d"]):
        by.setdefault(c_, []).append(float(x))
    C["reads"].append([float(np.clip(np.mean(by[k]), -1, 1)) for k in sorted(by)])
    C["Ds"].append(mt["D"])
    hm = H.harm(H.paired(vc, vp, mt["ids"]))
    hn = []
    for x in nulls_p:               # parent redraws (4 samples, own pair) vs the fresh parent read: the no-harm bound on the same rows
        vx = _four_votes(H, votes, four, x) or votes.get(x["name"]) or {}
        cov = [i for i in mt["ids"] if vx.get(i)]
        if cov:
            hn.append(H.harm(H.paired(vx, vp, cov)) * len(mt["ids"]) / len(cov))
    thr = eproc.harm_threshold(hn)
    C["harm"] += hm; C["harm_thr"] += (thr if thr is not None else float("inf"))
    C["rows"] += mt["rows"]; C["sumd"] += float(np.sum(mt["d"])); C["sumd2"] += float(np.sum(np.square(mt["d"])))
    C["up"] = C.get("up", 0) + mt["up"]; C["down"] = C.get("down", 0) + mt["down"]
    e = eproc.eprocess_rows(C["reads"])
    C["E"] = e["E"]
    Dacc = C["sumd"] / C["rows"]
    bandit_revise(R, r, m["name"], Dacc, C["rows"])
    zacc = C["sumd"] / np.sqrt(C["sumd2"]) if C["sumd2"] > 0 else 0.0
    dC = (rc["C1"] - rc["C0"]) / rc["C0"] if rc.get("C0") and rc.get("C1") else 0.0
    cost_ok = cost_rule(Dacc, dC, rc.get("novelty", 0), min(rc["delta"], Dacc - 1e-9) if Dacc > 0 else rc["delta"], R.c9_cfg)[0] if R.c9_cfg else True
    guards = dict(no_harm=C["harm"] <= C["harm_thr"], unseen_gain=Dacc > 0, up_down=C["up"] > C["down"], critic=rc["checks"].get("critic") in (True, "True"),
                  z=bool(rc["global_"] or zacc >= 1.96), cost=bool(cost_ok))
    ok = bool(e["E"] >= K / CARRY_ALPHA and all(guards.values()))
    if ok:
        C["status"] = f"admitted r{r}"
    elif len(C["Ds"]) >= 2 and C["Ds"][-1] <= 0 and C["Ds"][-2] <= 0:
        C["status"] = f"dropped r{r} (D <= 0 on two consecutive reads)"
    elif len(C["reads"]) >= CARRY_MAX_READS:
        C["status"] = f"dropped r{r} ({CARRY_MAX_READS} reads, E {e['E']:.2f} < {K / CARRY_ALPHA:.0f})"
    C["log"] = C.get("log", []) + [dict(round=r, rows=mt["rows"], D=mt["D"], p=mt["p"], up=mt["up"], down=mt["down"], E=e["E"], harm=hm, harm_thr=thr)]
    info = dict(path="carried", read=len(C["reads"]), rows=mt["rows"], D=mt["D"], p=mt["p"], up=mt["up"], down=mt["down"], E=e["E"], E_needed=K / CARRY_ALPHA,
                D_acc=Dacc, z_acc=float(zacc), rows_acc=C["rows"], guards=guards, admissible=ok, status=C["status"])
    return ok, info


def carry_out(R, r, recs, tracks):
    """Amendment 8: this round's member-track candidates that are not admitted, with D > 0 and rows up > down, are carried (the selecting read is not a
    bet; E starts at 1). At most carry_cap carried at once, by D."""
    if not R.c9.get("a8"):
        return []
    K = int(R.c9.get("carry_cap", 4))
    C = R.state.setdefault("c1109_carry", {})
    live = sum(1 for c in C.values() if c.get("status") == "carried")
    new = sorted([(tracks[m["name"]]["D"], m, row, rc) for m, row, rc in recs if m["name"] in tracks and not row.get("carried") and not rc["admissible"]
                  and tracks[m["name"]]["D"] > 0 and tracks[m["name"]]["up"] > tracks[m["name"]]["down"]
                  and not (m.get("gate") or {}).get("scope", "").startswith("cat:")], key=lambda t: -t[0])     # 1109c-final: a gate write is judged once on its scope + control, not carried
    out = []
    for D, m, row, rc in new:
        if m["name"] in C:
            continue
        if live >= K:
            R.say(f"r{r}: 1109c A8 {m['name']} qualifies for carry (D {D:+.4f}) but the cap {K} is full: not carried"); continue
        mm = {k: v for k, v in m.items() if not k.startswith("_")}
        rr = {k: row[k] for k in ("name", "round", "slot", "card_rows", "zx_score") if k in row}
        C[m["name"]] = dict(since=r, reads=[], Ds=[], rows=0, harm=0.0, harm_thr=0.0, sumd=0.0, sumd2=0.0, status="carried", select_read=dict(D=D), member=mm, row=rr)
        live += 1; out.append(m["name"])
    alive = [n for n, c in C.items() if c.get("status") == "carried"]
    R.say(f"r{r}: 1109c A8 carried to the next round: {alive} (new this round {out})")      # 1109c audit 11-09: all live carries, not only new ones
    return out


def rejudge(R, H, r, members, by):
    """Amendment 2 (fix 11-09): candidates of an earlier round that reached stage 2 while every null read was empty (state c1109_rejudge: [name, ...])
    are rebuilt from their own stored proposal, answered on this round's score + SELECT rows (cached rows are not asked again) and judged with this
    round's stage 2, beside its own candidates. Each name is re-judged once; nothing else about them changes."""
    names = list(R.state.pop("c1109_rejudge", None) or [])
    if not names:
        return carry_in(R, H, r, members, by)
    hist = {x["name"]: x for x in R.state.get("hist", []) if x.get("name")}
    rows = sorted(set(R.data.of("score")) | set(R.data.of("select")))
    R.no_id(rows, "re-judge")
    out = []
    for n in names:
        x = _rebuild(R, H, r, n, hist, by, "re-judge")
        if x is not None:
            x[1]["rejudge_of_round"] = x[1]["round"]
            out.append(x)
    if out:
        R.ev.evaluate_many([(m, rows) for m, _ in out])
        R.say(f"r{r}: 1109c re-judge (amendment 2) of {[m['name'] for m, _ in out]} on {len(rows)} score + SELECT rows")
    return out + carry_in(R, H, r, members, by)


def decide(R, H, r, rd, members, top, by, q_score):
    """§6.2 item 3 + A1 on R: every stage-2 candidate and every member's null draw; Holm across candidates; greedy accept with re-read; null void."""
    rung = R.c9["rung"]
    nulls = [(x, m) for m in members for x in null_members(R, H, m)]
    ctl = controls(top)
    four = four_map(R, H, members, [m for m, _ in top] + ctl)
    votes = R.votes(members + [m for m, _ in top] + [x for x, _ in nulls] + list(four.values()) + ctl)
    Rb = R.no_id([i for i in R.data.of("score") if i not in set(q_score)] + R.data.of("select"), "R")
    R.c9_lobo_cache = {}
    delta, cal = delta_now(R, H, members, votes, Rb)
    R.state["c1109_delta"] = delta
    counts = {}
    for rw in R.c9_hist.records():
        if rw.get("accepted") and rw.get("component"):
            counts[rw["component"]] = counts.get(rw["component"], 0) + 1
    base0 = R.fz("fuse", R.fuser.fuse, members, votes, R.data, R.fusion)
    def read(m, p, mem, rows, nl):
        return MC.stage2(H, R, mem, m, p, votes, rows, nl, rung, delta, R.c9_cfg, counts, card_ok=_card_ok(m, top), base=base0 if mem is members else None)
    recs, Rms = [], {}
    for m, row in top:
        if row.get("stage2_status") != "ok":
            row.update(c1109=dict(admissible=False, reason=f"stage 2 status {row.get('stage2_status')}"), would_accept=False)
            continue
        p = by[m["parents"][0]]
        Rm = [i for i in Rb if i not in lineage_cards(R, dict(m, card_rows=row.get('card_rows')))]
        rc = read(m, p, members, Rm, null_members(R, H, p))
        if row.get("carried"):        # amendment 8: a carried candidate is not ranked on this round's stage-1 draw, so its fresh read uses all score + SELECT rows
            Rm = [i for i in R.no_id(sorted(set(R.data.of("score")) | set(R.data.of("select"))), "carry") if i not in lineage_cards(R, dict(m, card_rows=row.get('card_rows')))]
        recs.append((m, row, rc)); Rms[m["name"]] = Rm
    nrec = []
    for x, p in nulls:
        Rm = [i for i in Rb if votes[x["name"]].get(i)]
        if len(Rm) < 150:
            R.say(f"  1109c null {x['name']}: {len(Rm)} R rows (< 150), not read")
            continue
        others = [y for y in null_members(R, H, p) if y["name"] != x["name"]]
        nrec.append((x, read(dict(x, parents=[p["name"]]), p, members, Rm, others or [x])))
        if MC.A6[0]:
            MC.a6_null(nrec[-1][1])
    adj = holm_bonferroni({rc["name"]: rc["p_boot"] for _, _, rc in recs}) if recs else {}
    mult = {}
    for m, row, rc in recs:
        mult[rc["name"]] = adj[rc["name"]] / max(rc["p_boot"], 1e-12)
        MC.decide_holm(rc, adj[rc["name"]])
    a7 = bool(R.c9.get("a7"))
    trip, frozen, steps0 = [], False, []
    if a7:
        trip, frozen = a7_apply(R, H, r, recs, nulls, by, votes, four, Rms, Rb, delta, steps0)
    left = sorted([(m, row, rc) for m, row, rc in recs if rc["admissible"]], key=lambda t: -t[2]["lb"])
    chosen, steps, at_accept = [], [], {}
    while left:
        m, row, rc = left.pop(0)
        chosen.append(m); row.update(accepted=True, accept_order=len(chosen)); at_accept[m["name"]] = rc
        steps.append(dict(step=len(chosen), accepted=m["name"], lb=rc["lb"], p_holm=rc["p_holm"], within_band=rc["within_band"]))
        stay = []
        if a7:                     # amendment 7: the member track does not depend on the pool; every admissible candidate is kept
            continue
        for x, rx, cx in left:
            p = by[x["parents"][0]]
            c2 = MC.stage2(H, R, members + chosen, x, p, votes, [i for i in Rb if i not in lineage_cards(R, dict(x, card_rows=rx.get('card_rows')))], null_members(R, H, p), rung, delta, R.c9_cfg, counts)
            MC.decide_holm(c2, min(1.0, mult[cx["name"]] * c2["p_boot"]))
            steps[-1].setdefault("reread", []).append(dict(name=x["name"], admissible=c2["admissible"], reason=c2["reason"][:200]))
            if c2["admissible"]:
                stay.append((x, rx, c2))
            else:
                rx["dropped_after"] = m["name"]
        left = sorted(stay, key=lambda t: -t[2]["lb"])
    null_adm = [rc["name"] for _, rc in nrec if rc["admissible"]]
    void = None
    if a7:                         # amendment 7: nulls calibrate, never veto; only the A6' tripwire (a null's member-track gain > delta) voids
        steps += steps0
        for _, rc in nrec:
            rc["pool_admissible"] = rc["admissible"]
        null_adm = list(trip)
        if frozen:
            for m, row, rc in recs:
                if row.get("accepted"):
                    row.update(accepted=False, frozen=True)
            R.say(f"r{r}: 1109c A7 C2 monitor FROZEN: no admission this round ({[m['name'] for m in chosen]} not admitted); sigma/delta re-estimated on fresh draws")
            steps.append(dict(rule="1109c a7 C2 freeze", frozen=[m["name"] for m in chosen]))
            chosen = []
            ep = R.state.setdefault("c1109_epoch", {})
            for x, p in nulls:
                ep[p["name"]] = int(ep.get(p["name"], 0)) + 1
    if MC.A6P[0] and null_adm and not a7:     # amendment 6': a cost-path null admission voids nothing (logged); the epoch of its parent still moves
        cost_ = [rc["name"] for _, rc in nrec if rc["admissible"] and MC.null_path(rc) == "cost"]
        if cost_:
            R.say(f"r{r}: 1109c A6' cost-path null admission(s) {cost_}: logged, void nothing")
            steps.append(dict(rule="1109c a6p cost-path null", nulls=cost_))
            ep = R.state.setdefault("c1109_epoch", {})
            for x, p in nulls:
                if x["name"] in cost_:
                    ep[p["name"]] = int(ep.get(p["name"], 0)) + 1
        null_adm = [n for n in null_adm if n not in cost_]
    if null_adm:
        if MC.A6[0] and MC.A6B[0] and not a7:           # amendment 6 (b): only the admits that do not beat every admissible null are void
            void, kept, a6t = MC.a6_void([at_accept[m["name"]] for m in chosen], [rc for _, rc in nrec], seed=r)
            steps.append(dict(rule="1109c a6 dominance", nulls=null_adm, voided=void, kept=kept, tests=a6t))
            R.say(f"r{r}: 1109c A6: admissible nulls {null_adm}; kept {kept} (beat every one), void {void}; " + "; ".join(f"{n} vs {t['null']}: p {t['p']:.4f} dce {t['dce']:+.4f} dacc {t['dacc']:+.4f} rows {t['rows']} -> {'beats' if t['ok'] else 'no'}" for n, L in a6t.items() for t in L if t.get('rows')))
        else:
            void = [m["name"] for m in chosen]
        for m, row, rc in recs:
            if row.get("accepted") and m["name"] in void:
                row.update(accepted=False, void=True)
        chosen = [m for m in chosen if m["name"] not in void]
        ep = R.state.setdefault("c1109_epoch", {})
        for x, p in nulls:
            if x["name"] in null_adm:
                ep[p["name"]] = int(ep.get(p["name"], 0)) + 1
        steps.append(dict(rule="1109c void", nulls=null_adm, voided=void))
        R.say(f"r{r}: 1109c NULL ADMISSIBLE {null_adm}: {'A6 void' if MC.A6[0] and MC.A6B[0] else 'void'} {void}; delta is re-measured on fresh draws next round")
    for m, row, rc in recs:
        row["c1109"] = MC.public(rc)
        row["would_accept"] = bool(rc["admissible"])
        R.say(f"  1109c s2 {rc['name']}: rows {rc['rows']} global {rc['global_']} dS {rc['dS']:+.4f} (delta {delta:.4f}) p {rc['p_boot']:.4f} holm {rc['p_holm']:.4f} lb {rc['lb']:+.4f} up/down {rc['up']}/{rc['down']} fired {rc['fired']} z {rc['z']['z']} harm {rc['harm']:.1f}/{rc['harm_thr']} -> {'ADMIT' if rc['admissible'] else 'no'}: {rc['reason'][:200]}")
    for x, rc in nrec:
        R.say(f"  1109c NULL {rc['name']}: rows {rc['rows']} dS {rc['dS']:+.4f} lb {rc['lb']:+.4f} -> {'ADMISSIBLE (void)' if rc['admissible'] else 'no'}: {rc['reason'][:160]}")
    # History (RRSI): one record per edit (component = own op)
    for m, row, rc in recs:
        comps = row.get("own_ops") or m.get("own_ops") or m.get("ops") or []
        R.c9_hist.append_candidate(r, m["name"], [dict(id=f"C{k + 1}", component=c, hypothesis=(m.get("why") or "")[:300]) for k, c in enumerate(comps)],
                                   "ACCEPTED" if row.get("accepted") else "REJECTED", rc["dS"], (rc["C1"] - rc["C0"]) / rc["C0"] if rc["C0"] and rc["C1"] else 0.0,
                                   bool(row.get("accepted")), rc["S1"], rc["C1"], None, rc["reason"][:300])
    prov, null_prov = [], []
    if void is None:               # A3: scoped members that miss the full J1 but pass the provisional line are carried, without a vote
        for m, row, rc in recs:
            if not row.get("accepted") and not row.get("carried") and provisional_ok(rc):
                prov.append(m); row["c1109_status"] = "provisional"
        for x, rc in nrec:
            if provisional_ok(rc):
                null_prov.append(dict(x, parents=[rc["parent"]]))
        if prov or null_prov:
            R.say(f"r{r}: 1109c A3 provisional (no vote, read on later rotated SELECT): {[m['name'] for m in prov]}; nulls tracked {[x['name'] for x in null_prov]}")
    for m, row, rc in recs:
        row.setdefault("c1109_status", "admitted" if row.get("accepted") else "rejected")
        row["bench_specific"] = bool(rc.get("bench_specific"))
        if row.get("accepted") and rc.get("bench_specific"):
            m["bench_specific"] = True           # in the pool for ID; reported apart, never in the headline or skill credit
            R.say(f"  1109c {m['name']} admitted as bench_specific: {rc.get('bench_specific_why')}")
    rec_prov = dict(prov=prov, null_prov=null_prov)
    rec = dict(round=r, rung=rung, delta=delta, calibrate=cal, rows_R=len(Rb), provisional=[m["name"] for m in prov], null_provisional=[x["name"] for x in null_prov], stage2=[MC.public(rc) for _, _, rc in recs], nulls=[MC.public(rc) for _, rc in nrec],
               holm=adj, accepted=[m["name"] for m in chosen], void=void, null_admissible=null_adm, steps=steps)
    H.jdump(f"{rd}/c1109.json", rec)
    rec["_prov"] = rec_prov
    return chosen, steps, rec


def _four_votes(H, votes, four, x):
    pr = four.get(x["name"])
    return H.merge4(votes[x["name"]], votes[pr["name"]]) if pr is not None and votes.get(pr["name"]) else None


def a7_apply(R, H, r, recs, nulls, by, votes, four, Rms, Rb, delta, steps):
    """Amendment 7 (prereg_1109c 'a7'): primary admission on the member track (4 vs 4 samples, cluster sign-flip, Holm across the round's <= 3
    candidates); an off-scope-abstaining gate write keeps its pool read (A1/A2 decide_holm, fired-z + A3). The pool read of every other candidate is
    logged as secondary. Nulls: each parent redraw is read on the member track vs its parent; the A6' tripwire (null D > delta) voids the round's
    admits; the C2 binomial monitor over all null reads so far freezes acceptance for the round when P(X >= k | n, .05) < .05.
    Returns (tripwire null names, frozen)."""
    from scipy.stats import binom
    MC._CFG[0] = R.c9_cfg
    tracks, pvals, gv3 = {}, {}, {}
    carried = {}
    for m, row, rc in recs:
        p = by[m["parents"][0]]
        rc["pool_line"] = dict(dS=rc["dS"], p=rc["p_boot"], p_holm=rc["p_holm"], admissible=rc["admissible"], reason=rc["reason"][:300])
        if row.get("carried"):          # amendment 8: the e-process across rounds; not in this round's Holm family
            ok, info = carry_judge(R, H, r, m, row, rc, null_members(R, H, p), Rms.get(m["name"], []), votes, four)
            rc["a7"] = info; rc.update(admissible=ok, reason=("A8 carried e-process ADMIT" if ok else f"A8 carried: {info.get('status')}") + f" | pool line: {rc['pool_line']['reason'][:200]}",
                                       dS_admit=info.get("D_acc", 0.0))
            carried[m["name"]] = info
            continue
        vc, vp = _four_votes(H, votes, four, m), _four_votes(H, votes, four, p)
        Rm = Rms.get(m["name"], [])
        gate_write = bool(H.gate_scope(m) is not None or m.get("gate"))
        scope = [i for i in Rm if (vc or votes[m["name"]]).get(i)]
        own = [i for i in scope if own_scope(R, H, m, i)] if (m.get("gate") or {}).get("scope", "").startswith("cat:") else None
        pool_path = gate_write and len(scope) < len(Rm) and own is None
        rc["pool_line"] = dict(dS=rc["dS"], p=rc["p_boot"], p_holm=rc["p_holm"], admissible=rc["admissible"], reason=rc["reason"][:300])
        if own is not None:          # 1109c audit 11-09: a gate write is read on the member track on its own scope rows only (it was diluted by all rows)
            if vc is None or vp is None:
                rc["a7"] = dict(path="gate scope", missing_pair=True); pvals[m["name"]] = 1.0; continue
            mt = MC.member_track(H, R.data, vc, vp, own, seed=zlib.crc32(m["name"].encode()) % 100000)
            tracks[m["name"]] = mt; pvals[m["name"]] = mt["p"]
            if R.c9.get("gate_key") == "prompts":     # gate_v3: beat its matched control on the same rows and be above chance per gold value
                c = control_of(m)
                vx = _four_votes(H, votes, four, c) if c is not None else None
                if vx is None:
                    gv3[m["name"]] = dict(ok=False, why=f"no control read ({(m.get('gate_control') or {}).get('missing') or 'no control arm'})")
                else:
                    mc = MC.member_track(H, R.data, vc, vx, own, seed=zlib.crc32((m["name"] + "~ctrl").encode()) % 100000)
                    gok, gtab = gold_above_chance(R, H, vc, own)
                    gv3[m["name"]] = dict(ok=bool(mc["rows"] and mc["D"] > 0 and mc["p"] <= .05 and gok), vs_control=dict(rows=mc["rows"], D=mc["D"], p=mc["p"], up=mc["up"], down=mc["down"]),
                                          gold=gtab, above_chance=gok)
            continue
        if pool_path:
            pvals[m["name"]] = rc["p_boot"]
            rc["a7"] = dict(path="pool (off-scope-abstaining gate write)", scope_rows=len(scope))
            continue
        if vc is None or vp is None:
            rc["a7"] = dict(path="member", missing_pair=True)
            pvals[m["name"]] = 1.0
            continue
        rows = Rm if rc["global_"] else scope
        mt = MC.member_track(H, R.data, vc, vp, rows, seed=zlib.crc32(m["name"].encode()) % 100000)
        tracks[m["name"]] = mt
        pvals[m["name"]] = mt["p"]
    adj = holm_bonferroni(pvals) if pvals else {}
    for m, row, rc in recs:
        n = m["name"]
        if n in carried:
            R.say(f"  1109c A8 {n}: {carried[n]}")
            continue
        if n in tracks:
            mt = tracks[n]
            ok, why = MC.a7_decide(rc, mt, adj[n])
            if n in gv3 and not gv3[n]["ok"]:
                ok, why = False, list(why) + ["gate_v3: " + (gv3[n].get("why") or f"vs control {gv3[n]['vs_control']} above chance {gv3[n]['above_chance']}")]
            rc["a7"] = dict(path="gate scope" if n in gv3 or (m.get("gate") or {}).get("scope", "").startswith("cat:") else "member", rows=mt["rows"], D=mt["D"], p=mt["p"], p_holm=adj[n],
                            up=mt["up"], down=mt["down"], admissible=ok, why_not=why, gate_v3=gv3.get(n))
            rc.update(admissible=ok, reason=("A7 member track ADMIT" if ok else f"A7 member track no: {why}") + f" | pool line: {rc['pool_line']['reason'][:200]}",
                      dS_admit=mt["D"])
        elif rc.get("a7", {}).get("path", "").startswith("pool"):
            MC.decide_holm(rc, min(1.0, adj[n]))
            rc["a7"].update(p_holm=adj[n], admissible=rc["admissible"]); rc["dS_admit"] = rc["dS"]
        else:
            rc.update(admissible=False, reason="A7: no seed-paired read of candidate or parent")
        R.say(f"  1109c A7 {n}: {rc['a7']}")
    # nulls on the member track: tripwire + C2 tally
    tally = R.state.setdefault("c1109_a7_c2", dict(n=0, k=0, reads=[]))
    trip = []
    for x, p in nulls:
        vx, vp = _four_votes(H, votes, four, x), _four_votes(H, votes, four, p)
        if vx is None or vp is None:
            continue
        mt = MC.member_track(H, R.data, vx, vp, Rb, seed=zlib.crc32(x["name"].encode()) % 100000)
        if not mt["rows"]:
            continue
        tally["n"] += 1; tally["k"] += int(mt["p"] <= .05)
        tally["reads"].append(dict(round=r, null=x["name"], parent=p["name"], rows=mt["rows"], D=mt["D"], p=mt["p"]))
        if mt["D"] > delta:
            trip.append(x["name"])
        R.say(f"  1109c A7 null {x['name']} vs {p['name']}: rows {mt['rows']} D {mt['D']:+.4f} p {mt['p']:.4f}{' TRIPWIRE (D > delta)' if mt['D'] > delta else ''}")
    carry_out(R, r, recs, tracks)
    tail = float(binom.sf(tally["k"] - 1, tally["n"], .05)) if tally["n"] else 1.0
    frozen = tally["n"] > 0 and tail < .05
    steps.append(dict(rule="1109c a7", member_holm=adj, tripwire=trip, c2=dict(n=tally["n"], k=tally["k"], tail=tail, frozen=frozen)))
    R.say(f"r{r}: 1109c A7 C2 monitor: {tally['k']}/{tally['n']} null reads at p <= .05, P(X >= k) {tail:.4f}{' FREEZE' if frozen else ''}; tripwire {trip}")
    return trip, frozen



def _card_ok(m, top):
    row = next((rw for x, rw in top if x["name"] == m["name"]), {})
    return not ((row.get("card_rows_n") or 0) > 0 and (row.get("dacc_card") or 0) > 0 and (row.get("dacc_fit") or 0) <= 0)


# ------------------------------------------------------------------ e-process over later rounds (confirmation) ----------------------------------
PROV_Z = 1.28          # A3 provisional line (fable_1109d §1.3 item 3): z >= 1.28 with net > 0 (one-sided p ~ .10), plus every fixed J1 check except z and Holm
PAIR_DRAW = 3


A5_GLOBAL = [False]    # amendment 5: global members also take the provisional line (set from c1109 a5 at init)


def provisional_ok(rc):
    c = rc["checks"]
    return ((A5_GLOBAL[0] or not rc["global_"]) and all(c.get(k) for k in ("fired", "no_harm", "card_vs_unseen", "critic")) and rc["dS"] > 0 and rc["up"] > rc["down"]
            and rc["z"]["z"] is not None and rc["z"]["z"] >= PROV_Z and rc["z"]["net"] > 0)


def confirm_reads(R, H, r, members):
    """A3 sequential path, at round start after the rotation. Every tracked member (admitted, not yet confirmed; provisional, not in the pool; a tracked
    null) is re-answered on this round's SELECT rows inside its scope by its own answers and its seed-paired draw 3 (x4 samples); each fired
    picture-cluster is one bet of its e-process (persisted). confirmed (E >= 20): a provisional member joins the pool, an admitted one's claim is
    confirmed; removed (E < 1 after 3 reads, or 5 reads unconfirmed): it leaves the pool / the provisional list. A confirmed null voids the members
    confirmed this round and moves its parent's draws to the next epoch."""
    if not R.c9["eproc"]:
        return {}
    E = R.state.setdefault("c1109_eproc", {})
    PROV = R.state.setdefault("c1109_prov", {})
    sel = R.no_id(R.data.of("select"), "confirmation")
    live = [n for n, e in E.items() if e.get("status") == "open" and e.get("round") != r]
    if not live:
        return {}
    mem_of = {m["name"]: m for m in members}
    mem_of.update({n: x["member"] for n, x in PROV.items()})
    want = []
    for n in live:
        m = mem_of.get(n) or (E[n].get("member"))
        if m is None:
            continue
        want += [(m, sel), (H.draw_of(m, PAIR_DRAW), sel)]
    R.ev.evaluate_many(want)
    R.ev.wait(R.say)
    votes = R.votes(list(members) + [x for x, _ in want])
    out, drop, join, nulls_conf = {}, [], [], []
    for n in live:
        m = mem_of.get(n) or E[n].get("member")
        if m is None:
            continue
        rest = [x for x in members if x["name"] != n]
        if MC.NULLREF[0] and E[n].get("parent"):     # amendment 5: the sequential read's reference pool holds one parent redraw too (null_ref)
            pm_ = next((x for x in list(members) + [v["member"] for v in PROV.values()] if x["name"] == E[n]["parent"]), None) or H.REG.get(E[n]["parent"])
            ref_ = next((x for x in (null_members(R, H, pm_) if pm_ else []) if x["name"] != n), None)
            if ref_ is not None:
                rest = rest + [ref_]
                if ref_["name"] not in votes:
                    votes.update(R.votes([ref_]))
        bets = []
        for x in (m, H.draw_of(m, PAIR_DRAW)):
            rows = [i for i in sel if votes[x["name"]].get(i)]
            if not rows:
                continue
            rd_ = PS.read_R(H, R, rest, x, votes, rows)
            F = H.fired_rows(rd_["f0"].tab, votes[x["name"]], rd_["pa"], rows)
            bets += eproc.fired_cluster_d(rd_["ids"], rd_["clusters"], rd_["d_acc"], F)
        E[n]["reads"].append(bets)
        st = eproc.eprocess_rows_status(E[n]["reads"])
        E[n].update(status=st["status"], E=st["E"], round=r)
        out[n] = dict(E=round(st["E"], 3), status=st["status"], reads=st["reads"], bets=st["bets"], kind=E[n].get("kind"))
        if st["status"] == "confirmed":
            if E[n].get("kind") == "null":
                nulls_conf.append(n)
            elif E[n].get("kind") == "provisional":
                join.append(n)
        elif st["status"] == "removed" and E[n].get("kind") == "admitted":
            drop.append(n)
    if nulls_conf:              # void rule on the sequential path
        R.say(f"r{r}: 1109c A3: NULL CONFIRMED {nulls_conf}: this round's confirmations are void ({join}); parents' draws move to the next epoch")
        for n in join:
            E[n].update(status="void", E=1.0)
        join = []
        ep = R.state.setdefault("c1109_epoch", {})
        for n in nulls_conf:
            ep[E[n]["parent"]] = int(ep.get(E[n]["parent"], 0)) + 1
    if drop:
        R.state["members"] = [m for m in R.state["members"] if m["name"] not in drop]
        R.say(f"r{r}: 1109c e-process REMOVED {drop} from the pool")
    if join:
        R.state["members"] = R.state["members"] + [PROV.pop(n)["member"] for n in join]
        R.say(f"r{r}: 1109c A3: provisional members CONFIRMED and joined the pool: {join}")
    for n, e in E.items():
        if e.get("kind") == "provisional" and e.get("status") in ("removed", "void"):
            PROV.pop(n, None)
    R.say(f"r{r}: 1109c e-process: {out}")
    return out


def admitted_reads(R, r, chosen, rec, prov=(), null_prov=()):
    """Start the e-process of each admitted member, provisional member and provisional null at its first read (this round's bets on R)."""
    E = R.state.setdefault("c1109_eproc", {})
    byn = {x["name"]: x for x in rec["stage2"] + rec["nulls"]}
    for kind, L in (("admitted", chosen), ("provisional", prov), ("null", null_prov)):
        for m in L:
            rc = byn[m["name"]]
            st = eproc.eprocess_rows_status([rc["eproc"]["bets_read"]])
            E[m["name"]] = dict(reads=[rc["eproc"]["bets_read"]], status="confirmed" if st["status"] == "confirmed" else "open", E=st["E"], round=r, kind=kind,
                                within_band=rc["within_band"], weight=rc["weight"], parent=rc["parent"], member=dict(m) if kind != "admitted" else None)
            if kind == "provisional":
                R.state.setdefault("c1109_prov", {})[m["name"]] = dict(member=dict(m), round=r)


# ------------------------------------------------------------------ reflect: notes, one skill edit, policy challenger ----------------------------
REFLECT_SYS = ("You are the proposer of an evolving harness for a vision-language model, reflecting on one round. You see only measured records. "
               "Write procedural knowledge for yourself, never answers: no row ids, bench names inside rules, option letters with counts, or numbers tied to rows.")


def skills_used(H, props):
    """The 'skill' id a proposer named in its JSON answer (read from the stored answer file)."""
    for p in props:
        if "c1109_skill_used" in p:
            continue
        try:
            ans = H.parse_answer(json.load(open(p["file"])).get("content")) if p.get("file") else None
        except (OSError, ValueError, TypeError, H.Bad):
            ans = None
        p["c1109_skill_used"] = (ans or {}).get("skill") if isinstance(ans, dict) else None


def reflect(R, H, r, rd, props, kids):
    skills_used(H, props)
    if not R.c9["reflect"]:
        return {}
    lines = []
    for p in props:
        s1 = p.get("c1109_s1") or {}
        v = p.get("c1109") or {}
        verdict = "admitted" if p.get("accepted") else ("not admitted: " + ", ".join(k for k, ok in (v.get("checks") or {}).items() if not ok) if v else (p.get("stage2_by") or p.get("status") or "not read"))
        lines.append(dict(name=p.get("name"), ops=p.get("own_ops") or p.get("ops"), skill=p.get("c1109_skill_used"), valid=p.get("valid"), fired_on_FIT=s1.get("fired"),
                          net_rows_on_FIT=s1.get("net"), zx_FIT=None if p.get("zx_score") is None else round(float(p["zx_score"]), 2), verdict=str(verdict)[:200], why=str(p.get("why") or "")[:240]))
    sk = [dict(id=s["front"]["id"], status=s["front"]["status"], pattern=s["front"]["trigger"].get("pattern"), evidence={k: s["evidence"].get(k) for k in ("proposals", "passed_null")}) for s in R.c9_skills.latest()]
    mp = R.c9_meta.S
    user = ("## This round's proposals and verdicts (FIT numbers are on score rows; held-out verdicts are given without numbers)\n" + json.dumps(lines, ensure_ascii=False, indent=0)[:14000] +
            "\n\n## Your skills\n" + json.dumps(sk, ensure_ascii=False) +
            f"\n\n## Slot policy (program / cross / W-step slots of {R.a['slots']}): incumbent {mp['incumbent']}" + (f", challenger {mp['challenger']} on half the slots" if mp.get("challenger") else ", no challenger running") +
            "\n\nAnswer with one JSON block:\n```json\n{\"notes\": [\"2-3 short lines for your proposer_notes.md: what to try or avoid next and why\"],\n"
            " \"skill_edit\": null or {\"id\": \"snake_case\", \"new\": true, \"pattern\": \"one of prior_recitation, rule_inversion, missing_detail, commit_early, dispersion\", \"scope\": \"an input-only predicate name\", "
            "\"detector\": \"a predicate over your own trace, never the gold answer\", \"procedure\": [\"ordered steps, with an abstain branch\"], \"body\": \"<= 300 words: when it applies, what the output must contain, known failure modes, what to read first\"}"
            " or, to amend an existing skill, {\"id\": ..., \"edits\": [{\"op\": \"append\" or \"insert_after\", \"content\": ..., \"target\": ...}]},\n"
            " \"policy\": null or {\"program\": int, \"cross\": int, \"w\": int}  (a challenger slot split summing to " + str(R.a["slots"]) + "; only when none is running)}\n```")
    got = llm(R, rd).generate_many("reflect", [dict(id="reflect", system=REFLECT_SYS, user=user)])
    ans = H.parse_answer((got.get("reflect") or {}).get("content")) or {}
    out = dict(raw_ok=bool(ans))
    notes = [str(x)[:300] for x in (ans.get("notes") or [])][:3]
    hits = [n for n in notes if precheck(n)]
    notes = [n for n in notes if not precheck(n)]
    if notes:
        with open(R.c9_notes, "a") as f:
            f.write(f"\n### after round {r}\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    out.update(notes=notes, notes_dropped=len(hits))
    if ans.get("skill_edit"):
        st, info = R.c9_skills.write(ans["skill_edit"], r)
        out["skill"] = dict(status=st, **info)
    if ans.get("policy") and R.c9["meta_p"]:
        ok, why = R.c9_meta.propose(r, ans["policy"], why="reflector")
        out["policy"] = dict(proposed=ans["policy"], started=ok, why=why)
    H.jdump(f"{rd}/c1109_reflect.json", out)
    R.say(f"r{r}: 1109c reflect: {out}")
    return out


def round_end(R, H, r, rd, reqs, props, chosen, rec):
    """Skill lifecycle, P A/B bookkeeping, trajectory for the stall flag."""
    skills_used(H, props)
    used, passed = collections.Counter(), collections.Counter()
    by_slot = {q["slot"]: q for q in reqs if q.get("id") != "meta"}
    net = {}
    acc_names = {m["name"] for m in chosen}
    for p in props:
        q = by_slot.get(p.get("slot"), {})
        ids = set(q.get("c1109_skills") or [])
        if p.get("c1109_skill_used") in ids:
            used[p["c1109_skill_used"]] += 1
            if p.get("name") in acc_names and not p.get("bench_specific"):        # skill credit only for transferring members (fable_1109d patch 17)
                passed[p["c1109_skill_used"]] += 1
        if p.get("name") in acc_names:
            net[p["slot"]] = float((p.get("c1109") or {}).get("dS") or 0.0)
    B = [b["component"] for b in R.c9_hist.prune_set(r, R.c9_cfg.n_prune)]
    trans = R.c9_skills.lifecycle(r, dict(used), dict(passed), prune_ids=[])
    ab = R.c9_meta.record(r, {int(k): v for k, v in (R.state.get("c1109_sides", {}).get(str(r)) or {}).items()}, net) if R.c9.get("allocator") != "ts" else None
    band = bandit_update(R, r, reqs, props, None)
    R.state.setdefault("c1109_traj", []).append(float(rec.get("pool_S", 0.0)) if rec else 0.0)
    out = dict(skills_used=dict(used), skills_passed=dict(passed), skill_transitions=trans, prune_set=B, policy_ab=ab, policy=R.c9_meta.S.get("incumbent"), challenger=R.c9_meta.S.get("challenger"))
    out.update(q_log(R, r, reqs, props, acc_names))
    if band:
        out["slot_rows"] = band
    H.jdump(f"{rd}/c1109_round_end.json", out)
    R.say(f"r{r}: 1109c round end: {out}")
    return out


def q_log(R, r, reqs, props, acc_names):
    """Amendment 7 / fable_1109 §3: one row per proposal with its quality q (member-track mean paired gain if it reached stage 2, else stage-1 zx;
    both raw) and the A/B side it was written on; at the end of each 2-round window the one-sided Wilcoxon rank-sum (on > off) at alpha .10 decides
    whether the component stays on from r11."""
    by_slot = {q["slot"]: q for q in reqs if q.get("id") != "meta"}
    sides = (R.state.get("c1109_ab", {}) or {}).get(str(r), {})
    psides = (R.state.get("c1109_sides", {}) or {}).get(str(r), {})
    rows = []
    for p in props:
        q = by_slot.get(p.get("slot"), {})
        a7 = (p.get("c1109") or {}).get("a7") or {}
        q_s2 = a7.get("D") if a7.get("path") == "member" else ((p.get("c1109") or {}).get("dS") if a7 else None)
        q_s1 = p.get("zx_score")
        sd = sides.get(str(p.get("slot")), {})
        rows.append(dict(round=r, slot=p.get("slot"), side=sd.get("window"), op=p.get("op") or q.get("op"), used_notes=sd.get("notes"), used_skill_id=p.get("c1109_skill_used"),
                         ledger_on=sd.get("ledger"), skills_on=sd.get("skills"), policy_side=psides.get(str(p.get("slot")), psides.get(p.get("slot"))), q_s1=q_s1, q_s2=q_s2,
                         q=q_s2 if q_s2 is not None else q_s1, admitted=p.get("name") in acc_names, bench_specific=bool(p.get("bench_specific"))))
    R.state.setdefault("c1109_q", []).extend(rows)
    out = dict(q_rows=rows)
    win = ab_window(R, r)
    if R.c9.get("harness_ab") and win and ab_window(R, r + 1) != win:
        from scipy.stats import mannwhitneyu
        key = {"notes": "used_notes", "skills": "skills_on", "ledger": "ledger_on"}.get(win)
        W = [x for x in R.state["c1109_q"] if ab_window(R, x["round"]) == win and x.get("q") is not None and (x["round"] >= r - 1)]
        if key:
            a = [float(x["q"]) for x in W if x.get(key)]; b = [float(x["q"]) for x in W if x.get(key) is False]
        else:                       # P challenger window: policy sides from the existing meta-P A/B
            a = [float(x["q"]) for x in W if x.get("policy_side") == "ch"]; b = [float(x["q"]) for x in W if x.get("policy_side") == "inc"]
        pv = float(mannwhitneyu(a, b, alternative="greater").pvalue) if a and b else 1.0
        dec = dict(window=win, n_on=len(a), n_off=len(b), p=pv, on=bool(pv <= .10), round=r)
        R.state.setdefault("c1109_ab_dec", {})[win] = dec
        out["ab_decision"] = dec
        R.say(f"r{r}: 1109c A/B window {win}: on {len(a)} vs off {len(b)} proposals, one-sided Wilcoxon p {pv:.4f} -> {'ON' if dec['on'] else 'OFF'} from r11")
    return out


# ------------------------------------------------------------------ text pre-screen before a write (fable_1109d patch 6) --------------------------
PRESCREEN_NET = 3          # proceed only at net >= +3 rows over the scope's show rows (x4 paired samples), with no row down by >= 2 of 4 samples


def prescreen(R, H, r, rd, gated, by):
    """Before any engine time on a gate write: the note as a text prefix (a program member on the parent's genome) on the scope's SHOW rows, x4 samples
    (samples 0,1 + seed-paired draw 3), against the bare parent x4 on the same rows and seeds. A note that does not change answers as text is not written."""
    if not cfg(R).get("prescreen", True):     # amendment 11-09 (prereg_1109c.json "amendments"): Q3 failed, the screen is off; gate writes go straight to the judge
        if gated:
            R.say(f"r{r}: 1109c pre-screen off (amendment 11-09): {len(gated)} gate writes go to the judge unscreened")
        return list(gated)
    keep, recs = [], []
    for m, row in gated:
        note = (m.get("gate") or {}).get("note") or ""
        scope = (m.get("gate") or {}).get("scope") or ""
        key = scope[4:] if scope.startswith("cat:") else None
        rows = [i for i in R.data.of("show") if key is not None and H.vh.key(R.data.rows[i]) == key]
        if not rows:
            row.update(valid=False, status=f"invalid: pre-screen: scope {scope!r} has no show rows")
            recs.append(dict(name=row.get("name"), scope=scope, rows=0, passes=False)); continue
        parent = by[m["parents"][0]]
        prog = "def harness(row):\n    return {\"prefix\": " + json.dumps(note[:600]) + "}\n"
        pm = H.member(parent["genome"], prog, name=f"pre_{m.get('gtag', m['mid'])[:10]}", origin="prescreen", parents=[parent["name"]])
        bare = H.member(parent["genome"], None, name="bare")
        arms = [pm, H.draw_of(pm, PAIR_DRAW), bare, H.draw_of(bare, PAIR_DRAW)]
        R.ev.evaluate_many([(x, rows) for x in arms])
        recs.append((m, row, rows, arms, scope))
    if not recs:
        return keep
    R.ev.wait(R.say)
    out = []
    for x in recs:
        if isinstance(x, dict):
            out.append(x); continue
        m, row, rows, arms, scope = x
        v = R.votes(arms)
        def right(a, b, i):
            s = [bool(t.get("ok")) for vv in (v[a["name"]].get(i) or {}, v[b["name"]].get(i) or {}) for t in vv.values() if t is not None]
            return sum(s), len(s)
        diffs = []
        for i in rows:
            n1, c1 = right(arms[0], arms[1], i)
            n0, c0 = right(arms[2], arms[3], i)
            if c1 and c0:
                diffs.append(n1 - n0)
        up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        ok = (up - down) >= PRESCREEN_NET and not any(d <= -2 for d in diffs)
        rec = dict(name=row.get("name"), scope=scope, rows=len(rows), read=len(diffs), up=up, down=down, down_twice=sum(d <= -2 for d in diffs), passes=bool(ok))
        out.append(rec)
        row["prescreen"] = rec
        if ok:
            keep.append((m, row))
        else:
            row.update(valid=False, status=f"invalid: pre-screen: as a text prefix the note moved {up} rows up, {down} down ({rec['down_twice']} down by >= 2 of 4) on {len(diffs)} show rows of its scope; a write needs net >= +{PRESCREEN_NET} and none down twice")
    H.jdump(f"{rd}/c1109_prescreen.json", out)
    R.say(f"r{r}: 1109c pre-screen: {[(x['name'], x.get('up'), x.get('down'), x['passes']) for x in out]}")
    return keep
