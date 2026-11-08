# Multimodal evaluation track (selfedit)

Test sets, disjoint edit-selection pools, and a baseline harness for Qwen3.8-27B with images on vLLM.

Code: `~/selfedit/code/mm/` (Mac) = `/workspace/fudan/mingdazhang/selfedit/code/mm/` (Vesta).
Data: `/data/fd/mingdazhang/selfedit/data/mm/{sets,pool,images,raw,manifest.json}` (Vesta only).
Runs: `/data/fd/mingdazhang/selfedit/runs/mm/`.

## Files
- `prep_mm.py [DATA]` downloads the pinned HF revisions (recorded in `manifest.json`), saves the images, and writes
  `sets/<bench>.jsonl` and `pool/<bench>.jsonl`. `--only a,b` is a debug build into `DATA/debug/`; it never overwrites
  the official files.
- `gen_mm.py` is an async OpenAI client for vLLM. It sends images as base64 data URLs, takes n samples per item and
  appends one line per sample. Output is resumable. Each line holds `id, set, kind, answer, options, category, sample,
  reasoning, content, finish, tokens, prompt_tokens, secs`.
  - `--limit N` (or `--limits a=N,b=M`) takes the first N lines of each set. The sets are written in stratified
    order, so that is a stratified subset.
  - `--deadline-min M` stops starting new requests after M minutes. Sets run in the order given, so the early sets
    finish completely.
  - Without `--max-tokens`, vLLM uses max_model_len minus the prompt length. Image prompts vary in length, so a fixed
    cap could overflow the context.
- `grade_mm.py OUT.jsonl [--cats] [--md]` grades the samples. It reports pass@1, unbiased pass@k, the mean, p50 and
  p90 of completion tokens, and the mean prompt tokens. It also reports the truncated, looped, error and no-answer
  rates, the per-category pass@1 and the mean over categories, plus MMVP pair accuracy. As in `../grade.py`, a sample
  counts only if `finish == "stop"`.
- `run_mm.sh MODEL PORT MAXLEN OUT [gen_mm args]` serves with vLLM, runs `gen_mm.py`, grades, and stops the server.
  Pass extra vLLM flags in `$SERVE_ARGS`.

## Item format
`{id, bench, kind, prompt, image_paths, answer, options, category, group, img_sha256, img_dhash, ...}`. `kind` is one of:
- `mc`: the answer is a letter A–J.
- `num`: the answer is a number string. An optional `precision` (MathVista floats) means the prediction is rounded first.
- `yesno`: the answer is `yes` or `no`.
- `short`: a structured short string (BabyVision blanks, RealWorldQA words). It is compared as a set of `x-y` pairs or
  as a token tuple, after case, punctuation and filler words ("row", "column", "and") are removed.

Every prompt ends with a kind-specific instruction to put the final answer in `\boxed{}`. Grading takes the last
`\boxed{}` in the content. If there is none, it falls back to "answer is ...". For `mc`, an exact option text maps to
its letter. A prompt with exactly one `<image>` placeholder per image (ERQA) is interleaved. Otherwise the images come
first.

## Split rule (test vs pool)
1. Within a benchmark, items are grouped into connected components that share a source group (MMVP pair,
   HallusionBench figure set, NaturalBench sample, CharXiv figure), an image sha256, or a 256-bit dHash. Near-blank
   images have no dHash. Whole components are assigned to test or to pool, with a seeded and stratified draw (seed
   20261103). BLINK and CV-Bench use equal quotas per sub-task; the others are proportional.
2. Small single-split benchmarks (MMVP, V*, CountBench) are used whole as tests. Their pools come from a different
   source dataset of the same family: NaturalBench for MMVP, HR-Bench 4K for V*, PixmoCount val for counting.
3. Global pass: a pool item is dropped if any of its image sha256 or dHash values appears in ANY test set (across all
   benchmarks) or if its id is a test id. The drop counts are in `manifest.json` under `dropped`.
4. Each test file is in stratified interleaved order, so `--limit N` gives a stratified N-item subset.

## Datasets (HF revisions are pinned in `manifest.json`)
| bench | HF repo (split used) | license | answer format | role | test / pool |
|---|---|---|---|---|---|
| BabyVision | UnipatAI/BabyVision (train = the 388-item benchmark) | MIT | 135 MC + 253 blanks; 1 free-text blank dropped | split | 200 / 187 |
| ERQA | FlagEval/ERQA (test 400; mirror of github embodiedreasoning/ERQA) | CC-BY-4.0 (GitHub) | MC A–D, 1–4 images | split | 201 / 199 |
| BLINK | BLINK-Benchmark/BLINK (val 1901; test answers hidden) | Apache-2.0 | MC, 1–4 images, 14 sub-tasks | split, 30 per sub-task | 420 / 420 |
| CV-Bench | nyu-visionx/CV-Bench (test 2638 = 2D 1438 + 3D 1200) | Apache-2.0 | MC | split, 100 per task | 400 / 400 |
| MMVP | MMVP/MMVP (300 questions = 150 pairs) | MIT | MC (a)/(b) | full test | 300 / – |
| NaturalBench | BaiqiL/NaturalBench (1900 samples × 4 questions) | Apache-2.0 | yes/no + MC | pool for MMVP | – / 404 |
| V* Bench | craigwu/vstar_bench (191 = 115 attribute + 76 position) | MIT (GitHub) | MC | full test | 191 / – |
| HR-Bench 4K | DreamMr/HR-Bench (hr_bench_4k: 800 rows = 200 questions × 4 option cycles) | other (LICENSE file) | MC A–D | pool for V* | – / 200 |
| HallusionBench | lmms-lab-encoder/HallusionBench (= lmms-lab/…; image split 951, non-image 178 skipped) | BSD-3 (GitHub) | yes/no | split by figure set | 477 / 472 |
| CountBench | vikhyatk/CountBenchQA (491, images embedded) | not stated | integer 2–10 | full test | 491 / – |
| PixmoCount | allenai/pixmo-count (validation 540, Flickr URLs) | ODC-BY | integer | pool for counting | – / 50 in the baseline run (see notes) |
| RealWorldQA (guard) | xai-org/RealworldQA (765) | CC-BY-ND-4.0 | MC / yes-no / number / word | split | 401 / 364 |
| MMStar | Lin-Chen/MMStar (val 1500) | not stated | MC A–D | split | 401 / 398 |
| MathVista (guard) | AI4Math/MathVista (testmini 1000; test answers hidden) | CC-BY-SA-4.0 | MC + integer + float with precision (2 list/text items dropped) | split | 500 / 495 |
| CharXiv reasoning | princeton-nlp/CharXiv (val reasoning 1000; test answers hidden) | CC-BY-SA-4.0 | only numeric answer types 3 and 4 (450 items); text answers need an LLM judge and are skipped | split by figure | 231 / 218 |

Cross-benchmark image overlaps removed from pools: MMStar 2, MathVista 3, CharXiv 1. The other pools had no overlap.

### Deviations from the official protocols
- BabyVision officially uses an LLM judge on `\boxed{}` answers. Here the blank answers are graded by rules (number;
  coordinate or list tuple; set of `x-y` pairs).
- ERQA, BLINK, CV-Bench, BabyVision, HallusionBench, RealWorldQA, MMStar, MathVista and CharXiv are scored on a
  held-out half, not the full official set.
- MMVP and HallusionBench report per-question accuracy. MMVP pair accuracy is also printed (`pair_acc`).
- A uniform `\boxed{}` instruction replaces each benchmark's own prompt suffix.

## How to run
```bash
# data (CPU job; about 25 min, most of it Flickr downloads for PixmoCount)
gpu bg --name selfedit-mmprep "CUDA_VISIBLE_DEVICES= taskset -c 88-95 bash -c '. /workspace/fudan/mingdazhang/selfedit/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o \"^TRITON_[A-Z_]*\"); export HF_HOME=/data/fd/mingdazhang/selfedit/.hf; python -u -W ignore /workspace/fudan/mingdazhang/selfedit/code/mm/prep_mm.py'"
# baseline / edited model (GPU): tests first, pools second; interleaved so a deadline cut leaves stratified prefixes
gpu bg --name selfedit-mm-x --gpus 6 --mem 78 "bash /workspace/fudan/mingdazhang/selfedit/code/mm/run_mm.sh MODEL_DIR 8707 32768 OUT.jsonl \
  --sets $D/sets/*.jsonl --sets2 $D/pool/*.jsonl --limit 200 --interleave --n 4 --effort xhigh --deadline-min 150 --hard-deadline-min 170"
# merge replicas / re-runs, then grade
python combine_mm.py --out MERGED.jsonl A.jsonl B.jsonl [--drop-sets ...] [--drop-error-items]
python grade_mm.py MERGED.jsonl --cats --md
```
### Official-protocol check (BabyVision, ERQA): `protocol_check.py`
Our BabyVision p@1 (44.7 on 80 test items) is far below Qwen's reported 65.7 ("Without CI"; the model card gives no
prompt for it), and ERQA is 58.4 vs 65.5. `protocol_check.py build` re-prompts the same baseline items the official way
(`DATA/protocol/bvoff/{babyvision,erqa}_{test,pool}.jsonl`: 80 + 15 and 77 + 14 items):
- BabyVision (UniPat-AI/BabyVision `babyvision_eval/evaluate_model.py`): image, then question [+ `\nChoices:\n(A) ...`]
  + `\nThink about the question and give your final answer in \boxed{Answer} format.`, no system prompt, reasoning
  on; official scoring is an LLM judge (GPT-5.2 or Qwen-Max) on the last `\boxed{}`. `protocol_check.py judge` uses the
  official judge prompt with Qwen3.8 itself (non-thinking, temperature 0) as the stand-in judge, on both the new and
  the baseline outputs, so grader and prompt effects separate.
- ERQA (embodiedreasoning/ERQA `eval_harness.py`): original question ending in "Please answer directly with only the
  letter of the correct option and nothing else.", images at `visual_indices`, exact match after removing '.'.
```bash
python protocol_check.py build --base RUNS/q38_xhigh_n4.final.jsonl --out $D/protocol
P=$D/protocol/bvoff; POST_CMD="python $SE/code/mm/protocol_check.py judge --server http://127.0.0.1:8707/v1 \
  --items $P/babyvision_test.jsonl $P/babyvision_pool.jsonl --in OUT.jsonl RUNS/q38_xhigh_n4.final.jsonl" \
  bash run_mm.sh MODEL_DIR 8707 32768 OUT.jsonl --sets $P/babyvision_test.jsonl $P/erqa_test.jsonl \
  --sets2 $P/babyvision_pool.jsonl $P/erqa_pool.jsonl --n 4 --effort xhigh --interleave
python protocol_check.py report --base RUNS/q38_xhigh_n4.final.jsonl --official OUT.jsonl
```
`run_mm.sh` runs `$POST_CMD` after grading while the server is still up.

A second replica on another GPU should get disjoint sets (for example the pools), with its own output file. For a
62 GiB slice use `SERVE_ARGS="--max-num-seqs 16"` (KV cache about 55k tokens).

## Baseline: Qwen3.8-27B, thinking on, reasoning_effort xhigh, n=4, max_model_len 32768 (2026-11-03)
Sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence penalty 0, repetition penalty 1.0. Served by vLLM
0.33.0 on one H100 (GPU 6, util 0.925); a GPU 3 replica (62 GiB) ran pool items.

Files in `/data/fd/mingdazhang/selfedit/runs/mm/base/`:
- `q38_xhigh_n4.final.jsonl` holds the merged records. `q38_xhigh_n4.final.grade` holds the per-set JSON with categories.
- The raw records are `q38_xhigh_n4.jsonl` (GPU 6), `q38_xhigh_n4_mathvista_fix.jsonl` and `q38_xhigh_n4_pools_gpu3.jsonl`.

The GPU budget was ~3 h, so each set is the first ~80% of a stratified target, not the whole test file: 100 for most
sets, 168 for BLINK, 120 for CV-Bench, 60 for the guards. Items = items with all 4 samples.

| set | items | pass@1 | pass@4 | tok mean | tok p90 | trunc % |
|---|---|---|---|---|---|---|
| BabyVision | 80 | 44.7 | 63.7 | 8395 | 16448 | 0.3 |
| ERQA | 77 | 58.4 | 72.7 | 4621 | 10913 | 0.3 |
| BLINK | 134 | 75.2 | 87.3 | 3416 | 9187 | 0 |
| CV-Bench | 95 | 90.8 | 93.7 | 783 | 1496 | 0 |
| MMVP | 80 (40 pairs) | 83.8 (pair acc 68.8) | 91.2 | 961 | 2858 | 0 |
| V* | 80 | 87.2 | 95.0 | 667 | 1132 | 0 |
| HallusionBench | 79 | 83.9 | 96.2 | 2213 | 6218 | 0 |
| CountBench | 79 | 96.5 | 98.7 | 542 | 670 | 0 |
| RealWorldQA (guard) | 48 | 85.9 | 89.6 | 1476 | 2813 | 0.5 |
| MMStar | 48 | 81.8 | 85.4 | 1708 | 4080 | 0 |
| MathVista (guard) | 60 | 89.6 | 91.7 | 1553 | 3800 | 0 |
| CharXiv-num | 48 | 74.5 | 87.5 | 3729 | 10370 | 1.0 |
| pool/BabyVision | 15 | 30.0 | 53.3 | 8800 | 13699 | 0 |
| pool/ERQA | 14 | 66.1 | 85.7 | 4072 | 10515 | 0 |
| pool/BLINK | 20 | 96.2 | 100 | 2238 | 5383 | 0 |

No looping was detected anywhere, and fewer than 0.5% of samples had no parsable answer.

Weakest categories (pass@1, items):
- BLINK: Forensic_Detection 52.5 (10), Semantic_Correspondence 55.0 (10), IQ_Test 55.6 (9), Visual_Similarity 61.1 (9),
  Functional_Correspondence 66.7 (9), Relative_Reflectance 72.5 (10), Counting 75.0 (10).
- CV-Bench: 2D/Count 78.1, 2D/Relation 91.7, 3D/Depth 96.6, 3D/Distance 97.0.
- BabyVision by type: Visual Tracking 26.5 (17), Fine-grained Discrimination 43.4 (34), Spatial Perception 51.4 (18),
  Visual Pattern Recognition 65.9 (11).
- ERQA: Multi-view 32.1 (7), Spatial 42.2 (16), Task 43.8 (8), Trajectory 57.7 (13).
- HallusionBench VD/video 67.9 (14). MMStar fine-grained perception 59.4 (8).

### Notes and serving issues
- The `vllm serve` flags were `--reasoning-parser qwen3 --max-num-seqs 64 --max-num-batched-tokens 8192
  --limit-mm-per-prompt '{"image":16,"video":0}'`, without `--language-model-only`. Images work, interleaved ERQA
  prompts included.
- The baseline run used an image limit of 4. Three ERQA test items (6, 7 and 8 images; one more has 16) were
  rejected with HTTP 400 and are excluded above. `run_mm.sh` now sets 16.
- Throughput is bounded by the KV cache, not compute. With images the KV cache is 259k tokens (262k text-only). Long
  xhigh traces (BabyVision ~8–17k tokens) fill it with ~20 running requests, and generation falls to ~600 tok/s. The
  run averaged 1140 tok/s on GPU 6 and 330 tok/s on the 62 GiB GPU 3 replica. Across the n=4 samples the MM processor
  cache hit rate is 75% and the prefix cache hit rate is 15–50%.
- The only engine error (EngineDeadError) was at shutdown, after SIGTERM. FlashInfer JIT warnings appear at
  start-up.
- MathVista prompts from the first build had a stray "(Unit: nan)". The fix is in `prep_mm.py`, and MathVista was
  re-run (`q38_xhigh_n4_mathvista_fix.jsonl`); the old MathVista records are dropped in the merge.
- PixmoCount: 282 of the 540 Flickr URLs fail. Most of the images that do download have bytes that differ from the
  published sha256 (Flickr re-encodes them); they are kept with `sha_match` recorded.
- Cost at this throughput: all test files at n=4 ≈ 36M generated tokens ≈ 9 H100-hours. The pools are about the
  same again. 200 items per test set ≈ 24M tokens ≈ 6 H100-hours, of which ~9M are done.
