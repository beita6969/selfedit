# selfedit

Private working repo. Current main line (state of 2026-11-12): **training-free weight editing of Qwen3.8-27B, computed in closed form from the model's own reads and applied round by round.** No gradient training, no prompt text added at inference, reasoning effort medium throughout (effort is never an edit).

Data, weights, model reads, fit arrays and logs are not in the repo.

## Results

Held-out rows, used for reporting only (nothing is fitted, tuned or selected on them). Edited model = both accepted weight edits, **stock requests** (no added prompt, no image up-scaling). Brackets are 95 % intervals from a bootstrap over pictures. Gains are against one read of the stock model.

| | ID (5 benchmarks) | OOD (5 benchmarks) |
|---|---|---|
| Stock model, 1 read | 60.1 | 64.5 |
| Edited model, 1 read | 70.8 (**+10.7** [+8.7, +12.8]) | 69.2 (**+4.6** [+2.9, +6.3]) |
| Edited model, 2 reads + a third on disagreement | 72.5 (**+12.4** [+10.3, +14.8]) | 71.0 (**+6.4** [+4.6, +8.3]) |
| Same read policy for the stock model | +10.9 [+8.6, +13.4] | +5.0 [+3.2, +6.9] |

Cost of one read, as a multiple of one stock read: generated tokens x0.89 (ID) / x1.01 (OOD); total tokens with the prompt x1.57 / x1.47 (the finer image lattice lengthens the input).

Scope of these tables: ID = babyvision, blink, charxiv, visulogic, vlmsbiased; OOD = chartqapro, hallusionbench, perceptionbench, spatialviz, visualpuzzles. lego (ID) and erqa (OOD) were also read and are not in these tables; `v5/judge/heldout_perbench.py` prints the per-benchmark numbers for all twelve. Per-benchmark tables, the round table and the limits are in [`docs/results_1112.md`](docs/results_1112.md).

## How a weight edit is computed

A condition that makes the model answer correctly (a procedure in the prompt, a finer image lattice) is the teacher. The edit makes the model behave that way without the condition.

1. **High-dimensional signal.** One of the model's own reads is replayed token by token twice, with and without the condition. The difference of every block's output on the identical stream is the forcing: exactly what each block has to add for the base run to become the conditioned run (base + forcing = conditioned run, checked numerically). It is an exact target for KL(teacher ‖ student), not a first-order direction.
2. **Low-dimensional projection.** The part of the forcing that is a function of the stream state is fitted by a reduced-rank ridge read of the MLP input (one linear solve and one truncated SVD per block, a benchmark held out). Each rank-1 term rewrites the gate, read and write weights of one MLP neuron. No back-propagation, no training loop.
3. **Where to write, known before any fit.** The share of a condition's effect carried by the linear blocks' mixers predicts how much the units restore (six conditions: R = 0.21 + 0.70 x share, Pearson 0.97). A condition whose effect is row-specific image content is not a function of the state (R = -0.04) and has to be written at the visual interface instead.

## The two accepted edits

| | What changes | Size | In-distribution acceptance (JUDGE rows of the six ID benchmarks) |
|---|---|---|---|
| Edit 1, general | Vision patch-embedding kernel resampled to half the patch edge | 1 tensor | +8.67 ± 1.48, positive on every benchmark |
| Edit 2, ID-specific | The loop-evolved counting procedure compiled into gated MLP units | 4,160 units, 192 tensors | +9.73 ± 2.99 on its source benchmark, no benchmark negative |

One read on the JUDGE rows: stock 60.3 → edit 1: 69.0 → edits 1 + 2: 71.7.

## How it iterates

Edit 2 is not computed once. Each round starts from what the edited model still gets wrong:

1. The current model answers the rows itself, without the procedure.
2. On those reads, the forcing of the procedure is captured again (the residual).
3. The same units are refitted in closed form on the reads of all rounds (an update, not more units).
4. The step is chosen by a line search on held-out reads.
5. Stop when the held-out residual no longer falls.

| Round | Fall of the held-out residual from the fit | Gain on the 75 routed signal rows, no prompt text |
|---|---|---|
| 1 | — | +4.5 ± 2.0 |
| 2 | −27 % | +8.8 ± 2.5 |
| 3 | −7 % | +10.8 ± 2.6 |
| 4 | −4 % | +10.2 ± 3.2 (flat: stop; the artefact is round 3) |

For reference: the procedure sent as prompt text is +8.3; units with random read directions are between −0.5 and +0.8.

An always-on cell lengthened unrelated reads (it carried the procedure's line-by-line manner to every stream). A gate fitted on the stream state confines the units to the rows the procedure is for; with it, unrouted rows keep their accuracy and length.

## Acceptance protocol

Candidates are pre-screened on signal rows and accepted on JUDGE rows (sign-flip test over picture clusters with one benchmark left out, Holm); a candidate is never scored on its source benchmark for the general layer. Every measured effect has a prediction written before the reads and a random-direction null. Held-out rows are read only to report.

## Layout

| Path | Contents |
|---|---|
| `internals/` | The weight-edit computation: paired-pass capture of the forcing, closed-form fit, cell specs, gate, rounds, law table (see `internals/README.md`) |
| `v5/engine/` | Serving and reading engine: dispatcher, weight-cell workers, export of an edited checkpoint (`export_cell.py`, `wedit.py`), held-out reader (`hv_read.py`) |
| `v5/judge/` | v5 judge and the acceptance stages of the weight edits (`stage4_vp2m.py`, `stage5_w2.py`), held-out tables (`heldout_table.py`, `heldout_perbench.py`) |
| `v5/loop/` | v5 evolution loop (proposer, generations, signal rows) |
| `v5/py_op/` | Code-running read operator with its sandbox (request-level comparison system) |
| `v5/harness/`, `v5/ood/`, `v5/signal/` | Judges, OOD tooling, signal-row features |
| `selfedit/`, `launch/`, `baselines/` | Code of the earlier 1109c run, kept: `selfedit/rsi/vision_patch.py` and `mm_surgeon.py` are what edit 1 and the checkpoint export use |
| `ext_notices/` | Licences of vendored upstream code |

## Settings

- Generation: max_tokens 32768, reasoning effort medium, max_model_len 46000.
- Rows: learn 768 (signal 324, JUDGE 444), ID test 768, OOD v2 768; 128 rows per benchmark in each held-out set.
- Edited checkpoints are full exports; untouched tensors are bit-equal to the stock model (checked on two machines).

## Limits

- Edit 1 is the gain of a finer image lattice delivered through the weights; it matches request-level image up-scaling at fewer total tokens. It does not unlock an answer the model already held.
- Edit 2 is specific to counting-type rows; the OOD gain comes mainly from edit 1 and from reading more than once.
- A round is one command, but the step and the gate form are still chosen by rule by hand, and the weight versions are not yet wired into the loop's state machine.
- Cells derived from reasoning-effort templates in `internals/` are mechanism evidence only, not results.

Private collaboration repo. Ask before making it public.
