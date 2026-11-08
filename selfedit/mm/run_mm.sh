#!/bin/bash
# Serve one multimodal model with vLLM on the visible GPU(s), sample with gen_mm.py, grade, then stop the server.
# Usage: run_mm.sh MODEL_DIR PORT MAXLEN OUT [gen_mm args...]; extra vllm flags via $SERVE_ARGS (TP etc.), optional
# $WAIT_FILE (start sampling only once it exists) and $POST_CMD (run after grading, while the server is still up).
set -u
SE=/workspace/fudan/mingdazhang/selfedit
. $SE/venv/bin/activate
unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset $(env | grep -o '^TRITON_[A-Z_]*')
export MPLBACKEND=Agg HF_HOME=/data/fd/mingdazhang/selfedit/.hf VLLM_CACHE_ROOT=/data/fd/mingdazhang/selfedit/.vllm-cache
MODEL=$1 PORT=$2 MAXLEN=$3 OUT=$4; shift 4
mkdir -p "$(dirname "$OUT")"
UTIL=$(python -c "print(round(float('${GPU_MEM_FRACTION:-0.95}') - 0.05, 3))")
echo "[mm] $(date) model=$MODEL gpus=${CUDA_VISIBLE_DEVICES:-} mem_frac=${GPU_MEM_FRACTION:-}"
# Images on: no --language-model-only. ERQA has items with up to 16 images (BLINK up to 4); no video.
vllm serve "$MODEL" --served-model-name m --port $PORT --max-model-len $MAXLEN --gpu-memory-utilization $UTIL \
  --reasoning-parser qwen3 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image": 16, "video": 0}' ${SERVE_ARGS:-} > "$OUT.server.log" 2>&1 &
SRV=$!
for i in $(seq 1 360); do
  curl -sf localhost:$PORT/health > /dev/null && break
  kill -0 $SRV 2> /dev/null || { echo "[mm] server died"; tail -60 "$OUT.server.log"; exit 1; }
  sleep 5
done
curl -sf localhost:$PORT/health > /dev/null || { echo "[mm] server not ready after 30 min"; kill $SRV; exit 1; }
echo "[mm] $(date) server ready"
# Optional: wait for a data build to finish (e.g. WAIT_FILE=/data/fd/mingdazhang/gpu-logs/<prep job>.rc) before reading sets/images.
if [ -n "${WAIT_FILE:-}" ]; then until [ -f "$WAIT_FILE" ]; do sleep 10; done; echo "[mm] $(date) $WAIT_FILE present"; fi
python $SE/code/mm/gen_mm.py --servers http://127.0.0.1:$PORT/v1 --model m --out "$OUT" "$@"
python $SE/code/mm/grade_mm.py "$OUT" --cats | tee "$OUT.grade"
# Optional: more work against the live server before it stops (e.g. POST_CMD="python .../protocol_check.py judge ...").
if [ -n "${POST_CMD:-}" ]; then echo "[mm] $(date) POST_CMD"; eval "$POST_CMD"; fi
kill $SRV; wait $SRV
echo "[mm] $(date) done"
