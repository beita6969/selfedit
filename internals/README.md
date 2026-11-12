# internals: computing a weight edit from the model's own reads

Scripts of the weight-edit line, as run on the GPU server on 2026-11-12. They read cached reads of the model (traces files) and write captures, fit arrays, cell specs and tables; none of those outputs is in the repo. Paths at the top of each script point at the server layout and are overridden by `FI_*` environment variables.

## Steps and the scripts that do them

| Step | Scripts |
|---|---|
| Teacher-forced forwards with a manual layer loop, capture and patch hooks | `intlib.py` |
| Rows and cached reads for a condition (fit rows, held-out rows, unrouted rows) | `mk_rows_v.py`, `mk_rows_v2.py`, `mk_rows_c.py`, `mk_items3.py`, `mk_items4.py`, `mk_unr_bench.py`, `conv_*.py` |
| Fit-free measurements of a condition: size, attention-cut survival, mixer shares, depth profile | `a1_patch.py`, `a4_ctx.py`, `comp_sum.py`, `ana1.py` |
| Capture of the forcing (conditioned run minus base run on identical tokens) | `a3_inject.py`, `a3_inject2.py`, `procstat2.py` |
| Closed-form reduced-rank ridge read, row-held-out and leave-one-benchmark-out folds | `fit3p.py`, `fit_lobo.py`, `fit_lobo2.py` |
| Route key from the stream state, key capture, gate arrays | `rkey_fit.py`, `kcap.py`, `keyz_bench.py`, `mk_krt.py` |
| Neurons whose removal costs least (the slots the units take) | `nuse3.py` |
| Cell specs in the Surgeon edit format, random and rotated nulls | `mkproc.py`, `mkrand2.py`, `edits.py` |
| Evaluation of a cell in the forward pass (restoration R, dose, controls) | `b_fn.py` … `b_fn5.py`, `b_xb.py`, `chk_proc2.py` |
| One round of the iterated residual compile (prep, capture, refit on the aggregate, tables, specs) | `mk_round3.sh` (`mk_round.sh`, `mk_round2.sh` earlier forms), `ana_round.py` |
| Side effects on rows a cell is not for, by benchmark; gated cells by benchmark | `ana_leak.py`, `ana_leak_bench.py`, `ana_gate_bench.py` |
| Law table: one condition through the whole measurement | `chain_comp3.sh` (`chain_comp.sh`, `chain_comp2.sh`) |
| Finer image lattice as a condition (two tokenisations in one process), stack check | `pixlib.py`, `pixlib2.py`, `pixlib3.py`, `pix_run*.py`, `mk_rows_pix.py`, `ana_pix.py` |
| Hindsight test (is the right answer latent in the stream?) | `mk_rows_h.py`, `h_stats.py`, `h_ans.py`, `ana_h.py`, `ana_h2.py` |
| Late re-opening of the image path (closed by its stop rule) | `p2_reopen.py`, `p2_reopen2.py`, `ana_p2.py` |
| Which text component is worth compiling next, on signal rows | `entry_effects.py`, `entry_effects2.py` |
| Card worker and its task queue | `worker.sh`, `queue/`, `run_x.sh`, `run_c.sh`, `chain_*.sh` |

`note_py.txt` and `ao1.txt` are two of the conditions measured in the law table (the code-interface note and an always-on sentence).

## One round, in order

`mk_round3.sh <k> <export of the round k-1 model's reads> <arrays of round k-1> "<capture folders in the aggregate>"` writes the queue tasks of round k: prepare the reads, capture the forcing in shards, refit the same units on the aggregate of all rounds, evaluate the previous and the new cell on the previous model's held-out reads (pair and step tables), measure the leak on unrouted reads, build the specs with their nulls. The step is then chosen by the held-out residual, the gated form is built with `mk_krt.py` and `mkproc.py`, and the served artefact is the base checkpoint plus one cell (`v5/engine/export_cell.py`).

Scripts whose docstrings mention the xhigh or low effort templates were written for the earlier mechanism study; cells derived from effort templates are not results of this line.
