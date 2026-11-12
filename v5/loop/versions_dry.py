"""Dry driver for L1 proposer versions (versions.VersionsMixin): gen.Run with step_propose = v_step_propose and "l1" after "report".
gen.py itself is not changed (the lines to fold in are listed in the builder's report). Dry-only extras:
  --force-switch G     at generation G the first active challenger with a common window is switched in (decision forced)
  --dry-selfedit       the fake engine answers chats with proposer entries; a self-edit chat whose answer does not parse as an edit is
                       answered here by a deterministic edit drawn from the option list (so the self-edit path makes challengers)
  --fast-poll          poll the fake engine's files every 0.2 s
  --l1-rule am15|none, --l1-from G   the AM15 L1 gate (set on the args when gen.args lacks them)
usage: python3 versions_dry.py [--force-switch 2] [--dry-selfedit] [--fast-poll] <gen.py args>"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen                       # noqa: E402
import selfedit as SE            # noqa: E402
from versions import VersionsMixin   # noqa: E402

if "l1" not in gen.STEPS:
    gen.STEPS = gen.STEPS + ["l1"]


class VRun(*((gen.Run,) if issubclass(gen.Run, VersionsMixin) else (VersionsMixin, gen.Run))):
    force_switch = None
    dry_selfedit = False

    def __init__(self, a, force_switch=None, dry_selfedit=False, fast_poll=False):
        super().__init__(a)
        self.force_switch, self.dry_selfedit = force_switch, dry_selfedit
        if fast_poll:
            w = self.io.wait
            self.io.wait = lambda keys, deadline=None, every=0.2, label="", tick=300: w(keys, deadline, every, label, tick)
            self.lab_every = 0.2                        # AM18: the lab reads' close rule polls at the same interval

    def step_propose(self):
        return self.v_step_propose()

    def step_l1(self):
        return self.v_step_l1()

    def l1_rule(self, p, inc, ch):
        d, p = super().l1_rule(p, inc, ch)
        if self.force_switch == self.g and not getattr(self, "_forced", None):
            self._forced = ch
            return "switch", p
        return d, p

    def selfedit_answer(self, key):
        x = super().selfedit_answer(key)
        if not self.dry_selfedit or x is None:
            return x
        if SE.parse_selfedit(x.get("content") or "")[1] is None:
            return x
        st = self.v_state()
        opts = [o for o in self.blind_options(st["versions"][st["inc"]]) if o["type"] != "transfer"]
        o = random.Random(f"dryself|{key}").choice(opts)
        o = dict(o, why="dry self-edit")
        return dict(x, content="```json\n" + json.dumps(o) + "\n```")


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--force-switch", type=int, default=None)
    ap.add_argument("--dry-selfedit", action="store_true")
    ap.add_argument("--fast-poll", action="store_true")
    ap.add_argument("--l1-rule", choices=["am15", "am15perm", "none"], default=None,
                    help="L1 rule: am15 (Bayesian bootstrap, default), am15perm (permutation P for decisions), none (report only)")
    ap.add_argument("--l1-from", type=int, default=None, help="AM15 rule from this generation on (default versions.L1_FROM)")
    mine, rest = ap.parse_known_args(argv)
    a = gen.args(rest)
    for k in ("l1_rule", "l1_from"):                     # also when gen.args does not (yet) define them
        if getattr(mine, k) is not None or not hasattr(a, k):
            setattr(a, k, getattr(mine, k))
    VRun(a, force_switch=mine.force_switch, dry_selfedit=mine.dry_selfedit, fast_poll=mine.fast_poll).main()


if __name__ == "__main__":
    main()
