# selfedit 1109c: NEW adapter (§8.2 item 1). QwenLLM: the frozen stock Qwen as RRSI's / SkillOpt's LLM, through hboost Run.generate only (D2: no proxy).
# One prompts.json per phase; the gen worker's regime is fixed by the run (32768 tokens, effort medium, thinking on) and is never set here.
from __future__ import annotations

import json
import os

_CURRENT = [None]


def current():
    if _CURRENT[0] is None:
        raise RuntimeError("QwenLLM: no run bound (call QwenLLM(run, root).bind())")
    return _CURRENT[0]


class QwenLLM:
    def __init__(self, run, root):
        self.run, self.root, self.calls = run, root, []

    def bind(self):
        _CURRENT[0] = self
        return self

    def phase_dir(self, phase):
        d = os.path.join(self.root, phase)
        os.makedirs(d, exist_ok=True)
        return d

    def generate_many(self, phase, items):
        """items: [dict(id, system, user)] -> {id: answer dict(content, reasoning, finish, tokens)}. Restart-safe: an existing prompts.json of the phase is reused."""
        d = self.phase_dir(phase)
        f = os.path.join(d, "prompts.json")
        if not os.path.exists(f):
            with open(f + ".new", "w") as o:
                json.dump([dict(id=x["id"], system=x["system"], user=x["user"]) for x in items], o, ensure_ascii=False)
            os.replace(f + ".new", f)
        reqs = json.load(open(f))
        got = self.run.generate(reqs, d)
        self.calls.append(dict(phase=phase, n=len(reqs)))
        return {x["id"]: x for x in got}

    def generate(self, system, user, *, phase="single", max_tokens=32768, thinking=True):
        """Drop-in for rrsi.llm.generate / SkillOpt chat_optimizer: one prompt as its own phase. max_tokens and thinking are the run's (asserted, not set)."""
        assert max_tokens == 32768 and thinking, "1109c: the proposer regime is 32768 tokens, thinking on"
        k = sum(1 for c in self.calls if c["phase"].startswith(phase))
        out = self.generate_many(f"{phase}_{k:02d}", [dict(id="q0", system=system, user=user)])
        return (out.get("q0") or {}).get("content") or ""
