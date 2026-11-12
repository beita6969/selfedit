# Results of the weight-edit line (2026-11-12)

Model: Qwen3.8-27B, thinking mode, reasoning effort medium in every call. Edited model = edit 1 (vision patch-embedding kernel) + edit 2 (gated counting cell, round 3). Requests are the stock requests: no added prompt text, no image up-scaling.

Held-out rows are read for reporting only. Samples per system: stock 8, edit 1 alone 5, edits 1 + 2 seven (two machines); every sample is one read of every row, 0 errors, 0 length stops. Brackets: 95 % intervals, pictures resampled within benchmark. "Vote" = 2 reads and a third on disagreement, exact expectation over the samples.

Scope: the tables below cover 5 ID benchmarks (babyvision, blink, charxiv, visulogic, vlmsbiased) and 5 OOD benchmarks (chartqapro, hallusionbench, perceptionbench, spatialviz, visualpuzzles), 128 rows each. lego (ID) and erqa (OOD) were also read and are not in these tables. `v5/judge/heldout_perbench.py` prints all twelve benchmarks; with `DROP=1` it prints the five-benchmark macro numbers given here.

## 1. Macro averages

| System | ID | OOD | Generated tokens (ID / OOD) | Total tokens (ID / OOD) |
|---|---|---|---|---|
| Stock, 1 read | 60.06 | 64.51 | x1.00 | x1.00 |
| Stock, vote | 61.57 | 65.93 | x2.51 / 2.38 | x2.45 / 2.33 |
| Edit 1, 1 read | 68.38 (+8.32 [+6.4, +10.3]) | 68.00 (+3.49 [+1.8, +5.1]) | x0.88 / 0.96 | x1.56 / 1.44 |
| Edit 1, vote | 70.20 (+10.14 [+7.9, +12.4]) | 68.97 (+4.46 [+2.6, +6.3]) | x2.16 / 2.25 | x3.64 / 3.28 |
| **Edits 1 + 2, 1 read** | **70.78 (+10.72 [+8.7, +12.8])** | **69.15 (+4.64 [+2.9, +6.3])** | x0.89 / 1.01 | x1.57 / 1.47 |
| **Edits 1 + 2, vote** | **72.49 (+12.43 [+10.3, +14.8])** | **70.95 (+6.44 [+4.6, +8.3])** | x2.16 / 2.36 | x3.63 / 3.34 |

- Against the stock model under the same vote policy: ID +10.92 [+8.6, +13.4], OOD +5.02 [+3.2, +6.9].
- Increment of edit 2 over edit 1, one read: ID +2.41 [+1.0, +3.9]; OOD +1.15 [−0.1, +2.4].
- Against the request-level system of 11-11 (image floor 4096 + 11 text procedures), vote: ID −0.40 [−2.2, +1.6], OOD +0.70 [−1.0, +2.4]: level, with nothing added to the request.

## 2. Per benchmark

ID:

| Benchmark | Stock, 1 read | Edited, 1 read | Gain | Edited, vote | Gain | Of which edit 1 | Of which edit 2 |
|---|---|---|---|---|---|---|---|
| babyvision | 39.5 | 58.5 | +19.0 [+13.7, +24.6] | 61.6 | +22.2 [+16.5, +28.2] | +20.1 | −1.0 |
| vlmsbiased | 71.8 | 90.0 | +18.2 [+13.5, +23.8] | 91.2 | +19.5 [+14.5, +25.4] | +6.2 | +12.0 |
| visulogic | 36.1 | 42.9 | +6.7 [+2.5, +10.9] | 44.7 | +8.6 [+3.8, +13.4] | +7.1 | −0.4 |
| charxiv | 79.7 | 86.2 | +6.5 [+3.0, +10.0] | 88.2 | +8.5 [+5.0, +12.2] | +4.1 | +2.4 |
| blink | 73.2 | 76.5 | +3.2 [−0.7, +7.2] | 76.7 | +3.5 [−1.1, +8.1] | +4.1 | −0.9 |

OOD:

| Benchmark | Stock, 1 read | Edited, 1 read | Gain | Edited, vote | Gain | Of which edit 1 | Of which edit 2 |
|---|---|---|---|---|---|---|---|
| perceptionbench | 52.0 | 58.7 | +6.8 [+2.1, +11.9] | 61.0 | +9.1 [+4.0, +14.6] | +3.2 | +3.5 |
| chartqapro | 67.2 | 73.3 | +6.1 [+2.6, +10.1] | 74.6 | +7.4 [+3.8, +11.5] | +6.9 | −0.7 |
| hallusionbench | 73.4 | 79.0 | +5.6 [+2.3, +8.9] | 79.8 | +6.4 [+2.8, +10.1] | +5.3 | +0.3 |
| visualpuzzles | 67.9 | 71.0 | +3.1 [−0.3, +6.6] | 72.3 | +4.4 [+0.6, +8.3] | +2.4 | +0.7 |
| spatialviz | 62.1 | 63.7 | +1.6 [−1.3, +4.4] | 67.0 | +4.9 [+1.6, +8.2] | −0.4 | +2.0 |

"Of which" = one-read split: edit 1 alone minus stock, and edits 1 + 2 minus edit 1.

## 3. In-distribution acceptance (JUDGE rows, six ID benchmarks, one read)

| Step | Macro | Note |
|---|---|---|
| Edit 1 against stock | +8.67 ± 1.48 | positive on all six; accepted as general (p < 1e-4) |
| Edit 2 on top of edit 1 | +2.70 ± 1.11 | source benchmark +9.73 ± 2.99 (p = .0001), no benchmark negative; admitted as ID-specific; off-source +1.29 ± 1.19 (not general) |
| Routed / unrouted rows of edit 2 | +6.13 ± 2.36 / +1.75 ± 1.25 | unrouted tokens x1.013 |

Levels: stock 60.30 → edit 1: 68.98 → edits 1 + 2: 71.73.

Not admitted: the always-on round-2 cell (source benchmark +10.86 ± 2.79, but babyvision −6.8 ± 2.6 and unrelated reads 13 % longer).

## 4. Rounds of edit 2 (iterated residual compile)

| Round | Fitted on | Held-out residual KL on the student's own reads | Fall from the fit | 75 routed signal rows, no prompt text |
|---|---|---|---|---|
| 1 | reads generated with the procedure | 0.0282 | — | +4.5 ± 2.0 (1,040 units) |
| 2 | + reads of the unedited model | 0.0207 | −27 % | +8.8 ± 2.5 (step 1.25) |
| 3 | + reads of the round-2 model | 0.0197 (step 1.25) | −7 % | +10.8 ± 2.6 |
| 4 | + reads of the round-3 model | 0.0209 on the round-3 model's reads | −4 % | +10.2 ± 3.2 |

- Procedure as prompt text: +8.3. Random-read nulls: −0.5 to +0.8. Rank 96 / 128 instead of 64 changes the residual by 2–8 %: the limit is the distribution, not the rank.
- Fidelity of the round-1 cell: 0.65 on reads generated with the procedure, 0.55 on the unedited model's reads, 0.49 on the reads it generates itself, which is why the fit is repeated on the student's own reads.
- Gate: always-on at step 1.25 against gated at step 1.8, each minus its random null: routed +9.3 / +8.0, unrouted −1.6 / +2.2, unrouted tokens +20 % / −4 %.
- On the final base with four samples: gated round-3 cell minus its random null +7.3 ± 2.7 on routed rows (null against the base 0.0 ± 1.8).
- Round-4 gated forms against the round-3 gated cell on the same rows: +3.0 ± 1.8 and +3.3 ± 1.7. Below the pre-declared 2-standard-error rule at decision time, not JUDGE-tested, not counted.

## 5. Where to write: fit-free predictor against measured compile fidelity

| Condition | Linear-mixer share (before any fit) | Restoration R by units (held-out reads) |
|---|---|---|
| Code-interface note | 0.84 | 0.79 |
| Routed procedure g005c29 | 0.56 | 0.56 |
| Always-on sentence | 0.54 | 0.56 |
| Counting procedure g002c26 | 0.53 | 0.63 |
| Effort templates (mechanism evidence only) | 0.98 / 0.83 | 0.88 / 0.82 |
| Finer image lattice | 0.32 | −0.04 (row-specific content: interface write) |
| Stated gold answer (hindsight) | 0.73 | 0.47 / 0.62, but a stated wrong answer is restored better: only "an answer was stated" compiles |

Six clean rows: R = 0.21 + 0.70 x share, Pearson 0.97, mean error 0.05. Attention-cut measures do not predict (rank correlation 0.12–0.22).

## 6. Limits

- Edit 1 delivers the gain of a finer image lattice through the weights (level with request-level image up-scaling, fewer total tokens). No test found a latent right answer in the stream state (five independent negatives, the last one the hindsight row above).
- Edit 2 is specific to counting-type rows of the ID source benchmark; on OOD its increment is not significant.
- Step and gate form are chosen by rule by hand; the weight versions are not yet in the loop's state machine.
