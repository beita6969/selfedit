# selfedit 1109c SHIM (not SkillOpt code): replaces skillopt/model/__init__.py, whose backends (Azure, Claude, Codex, ...) are not used.
# SkillOpt's optimizer calls (clip.rank_and_select, meta_skill, slow_update) reach the frozen Qwen through harness.adapters.qwen_llm.
from __future__ import annotations


def chat_optimizer(system: str, user: str, max_completion_tokens: int = 16384, retries: int = 5, stage: str = "optimizer",
                   reasoning_effort: str | None = None, timeout: int | None = None) -> tuple[str, dict]:
    from harness.adapters.qwen_llm import current
    out = current().generate(system, user, phase=f"skillopt_{stage}")
    return out, {"stage": stage, "backend": "qwen_llm"}
