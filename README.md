# selfedit

This repo holds only the latest code of the current main run, **1109c-final** (`p1109c_final_c`, launched 2026-11-10 CST). Old versions, data, weights, answers and logs are not included. Synced from the working tree at commit `4516ca8`.

A Qwen3.8-27B population improves itself, training-free, through several kinds of edits: harness programs, image patch/resolution and image-token edits, verbosity steering, and gated last-layer (L63) weight writes. Accepted members are fused with Dawid-Skene.

## What 1109c-final adds

- **Dynamic slot allocation:** Thompson sampling over edit families (`harness/adapters/slot_bandit.py`, plan in `hw_round.py`) replaces the fixed slot quota. A floor slot per family, a cost-adjusted paired-gain reward, a grid prior and forgetting.
- **Two-stage judge:**
  - Stage 1 is a screen. Amendment 11: zx > 0, net > 0, fired ≥ 3, cap 3.
  - Stage 2 is a per-member paired test on the same seeds, with Holm; nulls calibrate.
  - A8: a cross-round anytime-valid e-process carry.
- **gate_v3 weight writes:**
  - Per-category keys are fit on unlabeled prompts.
  - Writer checks require recital ≥ .95 and misfire ≤ .01.
  - Each write is admitted only if it beats a matched off-category control note and is above chance for every gold value.
- **Bug fixes found in the 11-09 audit and dry runs:**
  - op_patch keeps the stock 16384 image-token cap; it was silently 4096.
  - The critic input is summarised, so it no longer overflows the 46k context, and a critic failure now fails closed.
  - Gate jobs run in the run's own tree.
  - The stop rule respects live carries.
  - Gate writes are judged on their own scope rows only.
  - The stock answers are copied, not symlinked.

## Layout

| Path | Contents |
|---|---|
| `selfedit/rsi/hboost.py` | Main driver: rounds, proposal, critic, answering, two-stage judge, pool |
| `selfedit/harness/` | 1109c judge, adapters (slot bandit, round plan), ops, frozen prereg |
| `selfedit/rsi/mm_gate.py`, `mm_selfrule.py`, `mm_surgeon.py` | Weight edits |
| `selfedit/rsi/vision_patch.py`, `mm_fovea.py` | Resolution / patch edits |
| `selfedit/rsi/vharness.py`, `orch_program.py` | Harness programs and genome build |
| `selfedit/rsi/res_fuse.py`, `fus_side.py` | Dawid-Skene fusion |
| `selfedit/mm/grade_mm.py`, `gen_mm.py` | Grader and generation |
| `selfedit/tests/` | Unit tests of the judge and adapters |
| `launch/` | Final prereg, amendments, dry run, grid, launch and boundary scripts |

## Settings

- Generation: max_tokens 32768, effort medium, max_model_len 46000.
- Pool: 6 IID benches × 256 rows (learn 128, ID 128), plus OOD v2. Headline metric is pass@1.

Private collaboration repo. Ask before making it public.
