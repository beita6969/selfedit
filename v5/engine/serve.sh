#!/bin/bash
# v4 E server: one vLLM OpenAI server per card, loopback only (127.0.0.1). v3 numerics (rsi/vharness.py TAIL + prereg gen):
# bf16 weights (auto), kv fp8, max_model_len 46000, chunked prefill 8192, gpu util .85, prefix cache off, images <= 16/prompt.
# usage: serve.sh PORT [MAX_NUM_SEQS] [extra vllm args...]   (card chosen by gpu bg --gpus, i.e. CUDA_VISIBLE_DEVICES)
SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit
. $SE/venv/bin/activate
unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset $(env | grep -o '^TRITON_[A-Z_]*')
export MPLBACKEND=Agg HF_HOME=$D/.hf VLLM_CACHE_ROOT=$D/.vllm-cache PYTHONUNBUFFERED=1
PORT=$1; MNS=${2:-64}; shift; shift
echo "serve start $(date +%s.%N) card=$CUDA_VISIBLE_DEVICES port=$PORT max_num_seqs=$MNS extra=$*"
exec vllm serve ${MODEL_DIR:-$D/models/Qwen3.8-27B} --served-model-name ${SERVED:-stock} \
  --host 127.0.0.1 --port $PORT \
  --tensor-parallel-size 1 --max-model-len 46000 --max-num-batched-tokens 8192 --enable-chunked-prefill \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 --max-num-seqs $MNS --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 40, "video": 0}' --reasoning-parser qwen3 \
  --seed 0 --disable-uvicorn-access-log "$@"
