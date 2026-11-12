# v4 engine API (builder → engine agent), 11-10

Purpose: the one interface between the v4 loop (builder) and the engines (engine agent). The loop never talks HTTP. It writes request batches and reads result files. Every engine (6 E servers, P, later W) sits behind the same protocol.

## 1. Fixed for every call (AM1)
- **Model:** stock Qwen3.8-27B, `/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B`, vLLM v0.33 (same venv as v3).
- **Engine:** max_model_len 46000; kv fp8; prefix cache **on** (router and parent prompts repeat); max_num_seqs 64; gpu_util .85.
- **Effort:** `reasoning_effort="medium"` with `enable_thinking=True` for mode `read` and mode `chat`. The router has thinking off (§3).
- **read sampling** (v3 `loop.SAMPLING`): temperature 1.0, top_p .95, top_k 20, min_p 0, presence 0, repetition 1.0. **max_tokens 32768** (the v3 prereg budget, one budget for all candidates). `temperature` may be overridden per request (a knob).
- **chat sampling (proposers):** the same SAMPLING with temperature .7, max_tokens 12000. Both may be overridden per request.
- **Images:** passed as file paths under /data/fd/mingdazhang/selfedit, in the order of the row's `image_paths`. Content layout is exactly `engine/surgery.py:image_content`: when the prompt has `<image>` placeholders matching the image count, text and images interleave; otherwise all images come first, then the text.

## 2. Files
Root `V4 = /data/fd/mingdazhang/selfedit/v4`:
- `rows/learn.jsonl`: the 768 learn rows (builder writes this file; fields as in iid_paper128_v3.jsonl plus `v4_role` ∈ show|judge|monitor). The engine side reads `prompt`, `image_paths`, `kind`, `answer`, `options`, `category`, `precision` from it, by `id`. The ID/OOD files are separate, and the loop never sends them during search.
- `q/in/<batch>.jsonl`: request batches written by the loop. Written to `<batch>.jsonl.tmp`, then renamed. One file is never modified after the rename.
- `q/out/<batch>.jsonl`: results, one line per finished request, appended as requests finish.
- `q/out/<batch>.done`: written when every request of the batch has a result (or was cancelled).
- `q/cancel/<batch>`: an empty file means "drop the not-yet-started requests of this batch". Started ones finish and are written.
  - E and T servers (dispatcher): queued requests of the batch are dropped and write NO line; in-flight ones finish (at most one read each).
  - W worker: `q/cancel/<batch>` cancels the whole batch and also aborts a running group of it; `q/cancel/<batch>@<edit_sha>` cancels only that edit's requests in the batch (checked every 5 s while a group generates; the edit is reverted and the worker moves on). Every cancelled W request gets a line `{"key", "error": "cancelled: <why>", "cancelled": true, "edit", "edit_sha"}`, so the batch completes.
  - The loop (AM17 b, from gen 6) treats a cancelled line, or a key that never gets a line after its batch was cancelled, as NOT READ: never as wrong, right or zero difference. It writes one batch per candidate (`g<g>_rung1_<cid>`, its W part `g<g>_rung1_<cid>_W`, and for a weight cell past its quarter `g<g>_rung1b_<cid>[_W]`) and cancels them whole.
- `engines.json`: engine agent's list of servers (host, port, card, role E|P|W, state).

**Idempotence:** a request whose `key` already has a line in any `q/out/*.jsonl` is not sent again. The engine side keeps an index. This is how both sides resume after a restart.

## 3. Request line (`q/in`)
```json
{"key": "g003|c017|judge|<row id>|0", "mode": "read" | "route" | "chat", "engine": "E" | "P" | "W" | "T", "priority": 0-9 (0 first),
 "row": "<row id>",                       // read, route
 "sample": 0,                              // read: sample index (seed = hash(key) when a seed is used)
 "prefix": "text" | null,                  // read: inserted as a text part right before the row's prompt text (the procedures of routed entries, joined by "\n\n"); null = stock prompt
 "settings": {"min_image_tokens": null | int, "max_image_tokens": null | int, "temperature": null | float},   // read: null = stock
 "condition": "text",                      // route only
 "messages": [{"role": "system"|"user", "content": "..."}], // chat only (text-only)
 "max_tokens": null | int, "temperature": null | float      // chat only overrides
}
```
- **read:** the executor answers the row. Image settings are applied by client-side resize to the token target, using the processor's smart_resize (patch 16, merge 2, i.e. 32 px per token side) and its resample. Image tokens are counted after resize. If per-request `mm_processor_kwargs` proves identical in token count, it may be used instead (engine agent's call, stated in its notes).
- **route (v1, 11-10; replaces the v0 wording, which gave 432/444 malformed replies in the mini):** user content = the row's stock content (images and prompt, the layout of a stock read, no prefix), then ONE more text part: `"Condition: <condition>.\nDo not solve the question above. Does this condition hold for it? Reply with exactly one word: yes or no."` (Venus pilot router_v2.py, variant v2). Thinking off (`enable_thinking=False`), temperature 0, max_tokens 3, `logprobs=True, top_logprobs=20`, `structured_outputs={"choice": ["yes", "no"]}`. Request field `"view"`: null = stock image view (the only one used now). Loop keys: `rt1|<cond16>|<row>`. The engine side caches by (row id, sha256(condition)[:16], view, "v1") in `V4/route_cache_v1.jsonl` and answers from the cache when present, including for later ID/OOD reports; the old v0 file route_cache.jsonl is never read for v1.
- **chat:** proposer calls (P engine).
- **priority:** 0 = proposer chat and router; 1 = candidate rung reads; 2 = parent cache for a new tip (JUDGE and SHOW); 5 = background (cache top-up, MONITOR). Higher-priority requests go to the front of the queue. Running requests are never pre-empted.

## 4. Result line (`q/out`)
```json
{"key": "...", "mode": "read", "row": "...", "sample": 0, "content": "...", "reasoning": "...", "finish": "stop" | "length",
 "tokens": 1234, "prompt_tokens": 2345, "image_tokens": [1024, 512], "ok": true, "loop": false, "grade_error": null,
 "server": "127.0.0.1:18400", "t_submit": 1794300000.0, "t_done": 1794300123.4}
```
- **read:** `ok` and `loop` are graded exactly as `rsi/mm_eval.py:_collect`. If finish ≠ "stop" or content is empty, `ok` = False; otherwise `ok` = `grade_mm.correct(r)[0]`. `loop` = `grade_mm.looped`. Graders come from `code-1109c/mm/grade_mm.py`, with exceptions counted as wrong. The record carries `id`, `kind`, `answer`, `options`, `category` and `precision` like v3's, so the loop can regrade.
- **route:** `{"key", "mode": "route", "row", "cond": "<hash16>", "view": null, "rv": "v1", "yes": true | false, "p": 0.0-1.0, "py", "pn", "p_src": "logprob" | "choice", "content": "yes", "cached": true | false}`. p = P(yes) / (P(yes) + P(no)) at the first generated token, summed over the top-20 tokens whose `strip().lower().strip(".,!")` is "yes" / "no" (raw logprobs); if neither appears, p = 1 or 0 from the constrained content (p_src "choice"). yes iff p >= 0.5.
- **chat:** `{"key", "mode": "chat", "content", "reasoning", "finish", "tokens"}`.
- **errors:** after 3 tries a line with `"error": "<text>"` and no content. The loop treats it as unread, never as wrong.

## 5. Throughput report (engine agent → coordinator and builder)
`V4/stats.jsonl`, one line per minute per server: running sequences, waiting, generated tok/s (1-min window), finished requests. The loop reads the last 10 minutes to size N (AM6).

## 6. Who builds what
- **Engine agent:** servers, the dispatcher daemon implementing §2-§5 (one process on Vesta, async HTTP, in-flight cap per server just under max_num_seqs, mixing all batches by priority), grading, route cache, stats, and the launch parent cache (submitted as ordinary `read` batches, tag `parent`, so the loop reads them like any result).
  - Keys for the launch cache: `p000|parent|<row>|<s>` with s = 0..3.
  - Batch file name: `p000_parent_<k>.jsonl`.
- **Builder:** rows/learn.jsonl (written before the cache read starts; done within 15 min), the loop, and the P/W in-process Surgeon workers when Stage 3 needs them.
  - Stage 1 sends proposer `chat` requests to engine "P". Until the Surgeon P exists, P may be one more vllm server on the P card with the same protocol. Stage 1 needs no proposer weight edits.

## 7. Stage 2 additions (agreed with the engine agent, 11-10)
- **W worker** (card 3, in-process Surgeon). It takes batches whose requests are all `"engine": "W"`; keep W requests in batches of their own.
  - Modes: read and chat (no route).
  - Read = the usual fields plus `"edit": {"op": "scale"|"skip"|"attn_temp"|"vision_scale", ...}` in the knob form of knobs.json "weights_stage2", or null for stock weights. The worker converts it with rsi/vharness.py `validate` (scope all) on the stock genome.
  - Requests are grouped by canonical edit JSON: one apply → generate → revert per group.
  - Result = usual fields plus `"edit"` and `"edit_sha"` (sha256 of the canonical JSON, first 16 chars).
  - A failed apply or revert check gives error lines, and the worker reloads.
  - The loop's W read keys carry the edit inside the prompt signature.
- **Chat images:** in mode chat, a message `content` may be a list of parts: `{"type": "text", "text": ...}` and `{"type": "image", "path": ..., "min_image_tokens": null|int, "max_image_tokens": null|int}`. Image parts are resized client-side with each part's own min/max. Chat results carry `"prompt_tokens"` on E and W.
- **W edit forms (engine agent, 11-10 16:45; w_worker.py `surgeon_edit`):**
  - `{"op": "scale"|"skip", "layer": L | "layers": [a, b] (inclusive) | [L1, L2, L3, ...] (explicit list, 3+ entries), "writer": "mlp"|"attn", "alpha": x}`: one vharness op per layer, applied and reverted together. skip has no alpha. vharness bounds hold: alpha .5–1.5, layers 0–63.
  - `{"op": "attn_temp", "first": a, "last": b, "alpha": x}` or with `"layers": [a, b]`. Query scale of the self-attention layers 3, 7, 11, … inside the range; alpha .5–2.
  - `{"op": "vision_scale", "alpha": x}`, alpha .7–1.4.
  - AM13 directions: `{"op": "amplify"|"project_out", "layer": L | "layers": [...], "writer": "mlp"|"attn", "direction": "<.npy|.pt under /data/fd/mingdazhang>", "dose": x}`.
    - Effect: W ← (I + dose·u uᵀ) W on the writer's output side (Surgeon amplify, side out). project_out means dose −1.
    - A 1-D vector [5120] applies the same direction to every layer of the band; a 2-D array [n_layers, 5120] gives one row per layer. Vectors are normalised by the worker. dose −8..8.
    - There is no "bias" op: Qwen's o_proj, out_proj and down_proj have no bias. A constant vector needs a gated unit (see "surgeon").
  - `{"op": "surgeon", "file": "<cells.json under /data/fd/mingdazhang>"}`: a JSON list of raw Surgeon primitive edits.
    - Each edit: `{"op": scale|add|set|rank1|amplify|project_out|zero_block, "param": HF name, selection keys head(s)/part/neuron(s)/rows/cols, "alpha"|"value"|"u"|"v"|"side"}`.
    - Array fields may be `{"npz": "<file relative to cells.json or absolute>", "key": k}` or `{"npy": ...}`.
    - Files referenced by an edit must never change afterwards, because edit_sha hashes the path, not the content.
    - Cells go under /data/fd/mingdazhang/selfedit/v4/cells/<name>/.
  - Pending W requests with the same canonical edit are merged across batches into one apply → generate → revert, up to 4,000 requests per group. Large batches are therefore not tail-bound.
- **Harness ops on read (E servers; dispatch.py `run_ops`):** optional `"ops"`, applied in the fixed order zoom → tiles → read → check. A missing field or [] gives the plain read.
  - `{"op": "zoom"}`:
    - Pass 1: the read content (prefix included), then the text part "Do not answer the question. Name the image region that matters most for answering it: reply with only [i, x0, y0, x1, y1], where i is the image number (1 for a single image) and x0 y0 x1 y1 are normalised coordinates between 0 and 1.". Thinking off, temperature 0, max_tokens 48.
    - The box is clipped to [0, 1], and each side is widened to at least 0.05.
    - The crop of the original file is resized (bicubic) to the source image's token count under the read's min/max settings, with a floor of 256. It is inserted right after the row's last image part, with no extra text.
    - If parsing fails, there is no crop and `zoom_fail` is true.
  - `{"op": "tiles", "n": 2|3}`: each image is split n×n, and every tile is resized to that image's token count (floor 256). Tiles are appended after the row's last image part, image by image, row-major.
  - `{"op": "check"}`:
    - One more call on the row content (prefix included, without crops or tiles), with the text part "A proposed final answer to this question: <last \boxed{} of the answer, or its last 300 characters>. Check it against the image(s) and the question. If it is wrong, correct it. Give your final answer in \boxed{Answer} format.".
    - Thinking on, effort medium, read sampling, same seed. The graded answer is the check call's answer.
  - Result: usual fields, plus `ops`, plus `op_trace` (per pass: op, tokens, prompt_tokens; zoom: reply, box, zoom_fail, zoom_tokens; read: tiles; check: proposed, ok_before, pred_before). `tokens` is the total over all passes.
- **pred on every read result** (E and W, with or without ops) = `grade_mm.correct(r)[1]` from code-1109c/mm/grade_mm.py, i.e. the extracted, normalised answer:
  - mc: the letter;
  - num: a float, or "a,b" for list golds;
  - yesno: "yes" / "no";
  - otherwise: the cleaned final answer text.
  - It is null when unfinished, empty, or on a grader error. For cached parents, compute `grade_mm.correct({...})[1]` on the same fields.

## 8. Tip weights: T servers (agreed builder ↔ engine agent, 11-10 ~20:30 CST, after t004 merged the first weight edit)
Why: once a tip holds a weight edit, every executor read of the tip needs stock + the tip's edits. Without T servers all of them go to the W worker (two cards) and the E cards idle.
- **Request fields (mode read; the loop sets them from gen 6):**
  - `"engine"`: `"E"` = stock weights, no edit; `"T"` = exactly a tip's weights (its `edit` equals that tip's full edit list); `"W"` = an edit beyond the tip's (a weight cell on top of the tip, or a stack holding one).
  - `"edit"`: unchanged meaning — the FULL ordered edit list applied on stock (one dict, or a list in entry order; absent / null = stock). Keys, the W worker's grouping and `edit_sha` are therefore unchanged.
  - `"edit_sha"`: sha256(json.dumps(edit, sort_keys=True, separators=(",", ":"))).hexdigest()[:16]; edit = null hashes to `74234e98afe7498f` (stock; e.g. pcAA test batches).
  - `"weights"`: `"stock"` or `"<tip id>"` whose edits are the leading part of `edit` (label only).
  - `"tip_n"`: how many leading items of `edit` are the tip's edits (0 for stock). A W worker may start from the T checkpoint and apply only `edit[tip_n:]`; today it applies the full list on stock.
- **Routing (dispatcher) by edit_sha:** an `"engine": "T"` request, and a gen-5-style `"engine": "W"` request whose edit_sha (sent, or computed from `edit` when absent) equals that of an up T server, goes to a T server with that edit_sha. If engines.json lists no up T server with that edit_sha, the request goes to the W worker as before (slow, never stalls). Several tips can share one weight set (a tip that only adds a text entry keeps its parent's weights), hence edit_sha and not the tip id.
- **engines.json T entry:** role `"T"`, `"weights": "<tip id>"`, `"edit_sha": "<16 hex>"`, `"model": "<served model name>"` (= tip id); the dispatcher sends `model` = that name.
- **T servers** are `vllm serve` on the exported checkpoint with the §1 settings; harness ops (§7 `run_ops`) and cancel work on them exactly as on E.
- **Batch ownership:** the W worker writes `.done` for a batch whose requests are all engine W or T (once every key has a line, whoever wrote it); the dispatcher writes `.done` for every other batch.
- **Stock stays stock:** proposer chat (P) and the router (route) always run on stock weights (one stock server is kept): the proposer lineage is separate, and route decisions must not change with the tip.
- **Tip files (from gen 6):** `tips/<id>.json` and `tip.json` also carry `"edits"` (the tip's ordered edit list exactly as it appears in requests: null, one dict, or a list) and `"edit_sha"` (null when there are no edits). The engine agent's exporter reads them and writes `/data/fd/mingdazhang/selfedit/v4/ckpt/<tip id>/` plus `ckpt/<tip id>/export.json` (edits, edit_sha, verification).

