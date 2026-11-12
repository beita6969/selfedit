#!/bin/bash
# Switch to T servers for tip TIP (edit_sha SHA): card 5 already serves the verified checkpoint (job selfedit-v4e-s5<tip>).
# 1. cards 1 2 4: stop the idle stock E servers, start T servers on the checkpoint (parallel)
# 2. W3/W7 (old code) stopped; engines.json: card 5 T up; dispatcher swapped (new code); W3/W7 restarted (new code: skip T-served, cancel)
# 3. cards 1 2 4 marked up as their servers answer
TIP=$1; SHA=$2
V4=/data/fd/mingdazhang/selfedit/v4; ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4
cd ~/selfedit/code-v4/engine_v4
setsrv() {  # port json-update
  gpu run "cd $V4 && python3 -c \"
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port']==$1: x.update(json.loads('''$2''')); x.pop('note',None) if x.get('role')=='T' and x.get('weights')!='stock' else None
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')\"" >/dev/null && echo "$(date +%H:%M:%S) $1 <- $2"
}
T="\"role\": \"T\", \"weights\": \"$TIP\", \"edit_sha\": \"$SHA\", \"model\": \"$TIP\", \"cap\": 40, \"max_num_seqs\": 96, \"gpu_util\": 0.92"
for c in 1 2; do
  setsrv 820$c "{\"state\": \"released\"}"
done
sleep 8
for c in 1 2; do
  for j in $(gpu jobs | awk -v c=$c '$1 ~ "^selfedit-v4e-s"c"(b|u[0-9]+(t[0-9]+)?)?$" && $2=="running" {print $1}'); do gpu stop $j | tail -1; done
  gpu bg --name selfedit-v4e-s$c$TIP --gpus $c "MODEL_DIR=$V4/ckpt/$TIP SERVED=$TIP bash $ED/serve.sh 820$c 96 --gpu-memory-utilization 0.92" | tail -1
  setsrv 820$c "{$T, \"state\": \"starting\", \"job\": \"selfedit-v4e-s$c$TIP\"}"
done
echo "$(date +%H:%M:%S) stop W3/W7 (old code) and the old dispatcher"
gpu stop selfedit-v4e-w3e | tail -1
gpu stop selfedit-v4e-w7 | tail -1
setsrv 8205 "{$T, \"state\": \"up\", \"job\": \"selfedit-v4e-s5$TIP\", \"started\": $(date +%s)}"
gpu stop selfedit-v4e-disp7 | tail -1
gpu bg --name selfedit-v4e-disp8 "SE=/workspace/fudan/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; export MPLBACKEND=Agg CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4; cd $V4 && taskset -c 84-95,180-191 python -u $ED/dispatch.py --stats-every 60" | tail -1
WCMD() { echo "SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=\$D/.hf VLLM_CACHE_ROOT=\$D/.vllm-cache; cd \$D/v4 && python -u $ED/w_worker.py --name $1 --util 0.92 --max-num-seqs 96"; }
gpu bg --name selfedit-v4e-w3f --gpus 3 "$(WCMD W3)" | tail -1
# W7 restarts after the stock floor test (card 7)
echo "$(date +%H:%M:%S) switch done; waiting for T servers 1 2 4"
left="1 2"
while [ -n "$left" ]; do
  sleep 15; nl=""
  for c in $left; do
    if gpu run "curl -sf 127.0.0.1:820$c/v1/models >/dev/null" >/dev/null 2>&1; then setsrv 820$c "{\"state\": \"up\", \"started\": $(date +%s)}"; echo "$(date +%H:%M:%S) card $c: T up"
    elif gpu jobs | grep -q "selfedit-v4e-s$c$TIP .*exited"; then echo "$(date +%H:%M:%S) card $c: T server FAILED"
    else nl="$nl $c"; fi
  done
  left=$nl
done
echo "$(date +%H:%M:%S) to_t done"
