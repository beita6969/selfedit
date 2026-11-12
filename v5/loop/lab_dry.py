"""Test driver for the proposer lab (S2-A): gen.Run with LabMixin mixed in, "lab" inserted between "propose" and "validate",
lab_turns set on the P0 props by slot from --lab-turns (a cycle, e.g. "1,2"), and step_validate / repair reading final_key(p).
gen.py itself is not changed; the builder folds these few lines into gen.py later:
  - STEPS gets "lab" after "propose"; class Run(LabMixin, ...);
  - step_validate: x = self.io.get(final_key(p)) (and the repair key derived from it);
  - repair: the follow-up turn's messages come from self.lab_request(final_key(p)) (a lab key is not in prompts_g###.jsonl).
usage: python3 lab_dry.py [--lab-turns 1,2] [--fast-poll] <gen.py args>"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen                     # noqa: E402
from lab import LabMixin, final_key   # noqa: E402

if "lab" not in gen.STEPS:
    gen.STEPS = gen.STEPS[:gen.STEPS.index("propose") + 1] + ["lab"] + gen.STEPS[gen.STEPS.index("propose") + 1:]


class LabRun(*((gen.Run,) if issubclass(gen.Run, LabMixin) else (LabMixin, gen.Run))):
    lab_pattern = (1, 2)

    def __init__(self, a, lab_pattern=None, fast_poll=False):
        super().__init__(a)
        if lab_pattern is not None:
            self.lab_pattern = tuple(lab_pattern)
        if fast_poll:                                   # dry runs only: poll the fake engine's files every 0.2 s instead of 5 s
            w = self.io.wait
            self.io.wait = lambda keys, deadline=None, every=0.2, label="", tick=300: w(keys, deadline, every, label, tick)
            self.lab_every = 0.2                        # AM18: the lab reads' close rule polls at the same interval

    def step_propose(self):
        super().step_propose()
        G = self.G()
        for p in G["props"] + G.get("props_p0", []):
            if p["author"] == "P0":
                p["lab_turns"] = self.lab_pattern[p["slot"] % len(self.lab_pattern)]

    def repair(self, props, tv):
        """gen.Run.repair with the follow-up messages looked up by self.lab_request (P0 file or lab file)."""
        reqs = []
        for p in props:
            q0 = self.lab_request(p["key"]) if p["author"] != "B" else None
            if q0 is None:
                continue
            x = self.io.get(p["key"])
            cand, err, fix = self.check(x, tv, p.get("card"))
            if err is None or not fix:
                continue
            msg = (f"Your answer could not be used: {err}.\nSend your entry again as exactly one json block in the answer format, "
                   f"within the limits listed above.")
            msgs = q0["messages"] + [{"role": "assistant", "content": (x or {}).get("content") or ""}, {"role": "user", "content": msg}]
            p["repair_key"] = p["key"] + "|fix1"
            p["repair_reason"] = err[:300]
            reqs.append(dict(key=p["repair_key"], mode="chat", engine="E", priority=0, messages=msgs, temperature=gen.PROP_TEMP, max_tokens=gen.PROP_MAX))
        if reqs:
            n = self.io.submit(f"g{self.g:03d}_chatfix", reqs)
            gen.log(f"  repair round: {len(reqs)} format- or naming-only failures get one follow-up turn ({n} new)")
            self.io.wait([q["key"] for q in reqs], self.deadline, label="repairs")
        return len(reqs)

    def step_validate(self):
        """gen.Run.step_validate on final_key(p): p["key"] is swapped for the duration of the call and restored after it."""
        G = self.G()
        orig = {id(p): p["key"] for p in G["props"] if p.get("key")}
        try:
            for p in G["props"]:
                if p.get("key"):
                    p["key"] = final_key(p)
            super().step_validate()
        finally:
            for p in G["props"]:
                if id(p) in orig:
                    p["key"] = orig[id(p)]
        by = {(p["author"], p["slot"]): p for p in G["props"]}
        for c in G["cands"]:
            p = by.get((c["author"], c["slot"])) or {}
            if p.get("lab_turns"):
                c.update(lab_turns=p["lab_turns"], lab=p.get("lab"), key_final=final_key(p) if p.get("key") else None)


if issubclass(gen.Run, LabMixin):        # gen.py has the lab folded in (final_key in step_validate and repair): use gen's own methods
    del LabRun.repair
    del LabRun.step_validate


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--lab-turns", default="1,2")
    ap.add_argument("--fast-poll", action="store_true")
    mine, rest = ap.parse_known_args(argv)
    LabRun(gen.args(rest), lab_pattern=[int(x) for x in mine.lab_turns.split(",")], fast_poll=mine.fast_poll).main()


if __name__ == "__main__":
    main()
