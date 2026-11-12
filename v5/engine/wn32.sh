#!/bin/bash
# test (a): same W-null rows/seeds on W3 capped at 32 running
V4=/data/fd/mingdazhang/selfedit/v4
WCMD="SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=\$D/.hf VLLM_CACHE_ROOT=\$D/.vllm-cache; cd \$D/v4 && python -u /workspace/fudan/mingdazhang/selfedit-v4/engine_v4/w_worker.py --name W3 --util 0.92 --max-num-seqs 32"
until gpu run "test -f $V4/q/out/wnull_t002_1.done" >/dev/null 2>&1; do sleep 60; done
t0=$(gpu run "date +%s" | tail -1)
echo "$(date +%H:%M:%S) wnull done; restart W3 at 32"
gpu stop selfedit-v4e-w3f | tail -1
gpu bg --name selfedit-v4e-w3g --gpus 3 "$WCMD" | tail -1
until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/w/W3.up.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 15; done
echo "$(date +%H:%M:%S) W3 up (32)"
gpu run "cd $V4/q/in && sed 's/\"key\": \"wnull|/\"key\": \"wn32|/' wnull_t002_1.jsonl > .wn32_t002_1.tmp && mv .wn32_t002_1.tmp wn32_t002_1.jsonl && wc -l wn32_t002_1.jsonl && head -c 200 wn32_t002_1.jsonl"
echo "$(date +%H:%M:%S) wn32 submitted"
