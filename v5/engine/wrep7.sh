#!/bin/bash
# after czero_t004_2: report, stop the cell server, W7 worker on card 7, submit wrep_c30_1
V4=/data/fd/mingdazhang/selfedit/v4; ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4
until [ "$(gpu run "grep -c . $V4/q/out/czero_t004_2.jsonl" 2>/dev/null | tail -1)" -ge 444 ] 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) czero complete"
until gpu run "grep -q '27ad64783a29e1bc: done' /data/fd/mingdazhang/gpu-logs/selfedit-v4e-c7b.log" >/dev/null 2>&1; do sleep 15; done
gpu run "tail -3 /data/fd/mingdazhang/gpu-logs/selfedit-v4e-c7b.log | cut -c1-200"
t0=$(gpu run "date +%s" | tail -1)
gpu stop selfedit-v4e-c7b | tail -1
WCMD="SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=\$D/.hf VLLM_CACHE_ROOT=\$D/.vllm-cache; cd \$D/v4 && python -u $ED/w_worker.py --name W7 --util 0.92 --max-num-seqs 96"
gpu bg --name selfedit-v4e-w7r --gpus 7 "$WCMD" | tail -1
until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/w/W7.up.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 15; done
echo "$(date +%H:%M:%S) W7 up"
gpu run "python3 $V4/tmp_g5r/mkwrep_engine.py go | tail -1"
echo "$(date +%H:%M:%S) wrep submitted"
