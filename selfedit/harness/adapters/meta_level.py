# selfedit 1109c: NEW adapter (§8.2 item 8, §6.3). MetaLevel: the P (operator policy) challenger on half the slots for 2 rounds, and the frozen-anchor
# guard (Self-HPO restore_infra pattern: the anchor is read-only and checked before every proposal phase; a change stops the run).
# The policy is the slot split {program, cross, w} (w = W-step slots, dealt over the admitted families); menu/grid slots are not in it (§4 'out').
from __future__ import annotations

import hashlib
import json
import os

ROLES = ("program", "cross", "w")
DEFAULT = dict(program=4, cross=2, w=6)          # §3.1: H half / W half


def tree_sha(paths):
    h = hashlib.sha256()
    for root in paths:
        for dp, dn, fn in sorted(os.walk(root)):
            dn.sort()
            for f in sorted(fn):
                if f.endswith((".pyc",)) or "__pycache__" in dp:
                    continue
                p = os.path.join(dp, f)
                h.update(os.path.relpath(p, root).encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


class Anchor:
    def __init__(self, harness_dir, state):
        self.paths = [os.path.join(harness_dir, "anchor"), os.path.join(harness_dir, "frozen")]
        self.state = state

    def check(self, where):
        s = tree_sha(self.paths)
        want = self.state.setdefault("anchor_sha", s)
        if s != want:
            raise SystemExit(f"1109c anchor changed before {where}: {s} != {want} (the anchor is read-only; restore it and restart)")
        return s


def valid_policy(p, slots):
    if not isinstance(p, dict) or set(p) != set(ROLES):
        return False, f"policy must have exactly {ROLES}"
    if any(not isinstance(p[k], int) or p[k] < 0 for k in ROLES) or sum(p.values()) != slots:
        return False, f"counts must be non-negative integers summing to {slots}"
    if p["program"] + p["cross"] < 2 or p["w"] < 2:
        return False, "at least 2 H slots and 2 W slots"
    return True, None


class MetaLevel:
    """state['meta_p'] = {incumbent, challenger, ab_from, ab_rounds, slots: {round: {slot: 'inc'|'ch'}}, results: [...], log}"""
    def __init__(self, state, slots, delta_slot_fn):
        self.S = state.setdefault("meta_p", dict(incumbent=dict(DEFAULT), challenger=None, ab_from=None, results=[], log=[]))
        self.slots, self.delta_slot = slots, delta_slot_fn

    def plans(self, r):
        """Per slot: which policy deals it ('inc' or 'ch'); 50/50 by slot parity while an A/B runs."""
        S = self.S
        if S["challenger"] and S["ab_from"] is not None and r < S["ab_from"] + 2:
            return {k: ("ch" if k % 2 else "inc") for k in range(self.slots)}
        return {k: "inc" for k in range(self.slots)}

    def policy(self, side):
        return self.S["incumbent"] if side == "inc" else self.S["challenger"]

    def propose(self, r, p, why=""):
        if self.S["challenger"]:
            return False, "an A/B is running"
        ok, e = valid_policy(p, self.slots)
        if not ok:
            self.S["log"].append(dict(round=r, rejected=p, why=e)); return False, e
        if p == self.S["incumbent"]:
            return False, "same as incumbent"
        self.S.update(challenger=dict(p), ab_from=r + 1, why=why)
        self.S["log"].append(dict(round=r, challenger=p, starts=r + 1, why=why))
        return True, None

    def record(self, r, side_of_slot, net_of_slot):
        """After round r: per side, the admitted members' summed dS (null-calibrated by the round's void rule) over its slots."""
        if not self.S["challenger"] or self.S["ab_from"] is None or not (self.S["ab_from"] <= r < self.S["ab_from"] + 2):
            return None
        tot = dict(inc=0.0, ch=0.0)
        for k, side in side_of_slot.items():
            tot[side] += float(net_of_slot.get(k, 0.0))
        self.S["results"].append(dict(round=r, **tot))
        if r == self.S["ab_from"] + 1:          # 2 rounds done: decide
            rs = [x for x in self.S["results"] if self.S["ab_from"] <= x["round"] <= r]
            ch, inc = sum(x["ch"] for x in rs), sum(x["inc"] for x in rs)
            d = float(self.delta_slot())
            win = ch >= inc + d
            self.S["log"].append(dict(round=r, decided="challenger" if win else "incumbent", ch=ch, inc=inc, delta_slot=d, policy=self.S["challenger"]))
            if win:
                self.S["incumbent"] = self.S["challenger"]
            self.S.update(challenger=None, ab_from=None)
            return dict(decided=True, win=win, ch=ch, inc=inc, delta_slot=d)
        return dict(decided=False, **tot)
