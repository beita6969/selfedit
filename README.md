# selfedit

This repo holds only the latest code that the final run hb/paper128b (2026-11-08) actually executes. Old versions, test scripts, data, weights, answers and logs are not included.

A Qwen3.8-27B population improves itself through several kinds of edits: weights, structure, input resolution, token budget and harness programs. The accepted members are then fused with Dawid-Skene.

## Layout

| Path | Contents |
|---|---|
| `selfedit/rsi/hboost.py` | Main driver: rounds, candidate proposal, answering, acceptance (FIT/SELECT, LB), plateau and combination tests |
| `selfedit/rsi/res_fuse.py`, `fus_side.py` | Dawid-Skene fusion (ds_agree) and side lines |
| `selfedit/rsi/mm_gate.py`, `mm_selfrule.py`, `mm_surgeon.py` | Weight edits: gated rule write and point edits |
| `selfedit/engine/surgery.py`, `rsi/arch_xf.py` | Weight and structure surgery (layers, heads, experts) |
| `selfedit/rsi/vision_patch.py`, `mm_fovea.py` | Resolution / patch edits |
| `selfedit/rsi/vharness.py`, `orch_program.py` | Harness programs |
| `selfedit/rsi/evo.py`, `self_propose.py`, `meta_propose.py`, `self_diagnose.py` | Lineage and self-proposal / self-diagnosis |
| `selfedit/rsi/res_queue.py`, `ck_worker.py` | Multi-card queue and card workers |
| `selfedit/mm/grade_mm.py`, `gen_mm.py` | Grader and generation |
| `baselines/` | SFT / GRPO baselines; same grader, generation settings and pool |
| `launch/` | Prereg, sha list, deploy and launch scripts |

## Settings

- Generation: max_tokens 32768, effort medium, max_model_len 46000, fp8 KV, seed 101.
- Pool: 6 benches × 256 rows (learn 128, ID 128).
- Code sha: every file matches the `code_shas` in `launch/prereg128b.json`.

Private collaboration repo. Ask before making it public.
