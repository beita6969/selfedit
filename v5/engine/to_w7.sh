#!/bin/bash
# card 7 E -> W7; restart W3 with the claim code (W3 idle)
cd ~/selfedit/code-v4/engine_v4
V4=/data/fd/mingdazhang/selfedit/v4
setst() { gpu run "cd $V4 && python3 -c \"
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port']==8207: x['state']='$1'
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')
print('card 7 ->','$1')\"" | tail -1; }
WCMD() { echo "SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=\$D/.hf VLLM_CACHE_ROOT=\$D/.vllm-cache; cd \$D/v4 && python -u /workspace/fudan/mingdazhang/selfedit-v4/engine_v4/w_worker.py --name $1 --util 0.92 --max-num-seqs 96"; }
t0=$(gpu run "date +%s" | tail -1)
echo "$(date +%H:%M:%S) W3 restart (claims)"
gpu stop selfedit-v4e-w3d | tail -1
gpu bg --name selfedit-v4e-w3e --gpus 3 "$(WCMD W3)" | tail -1
echo "$(date +%H:%M:%S) card 7 drain"; setst drain
until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/q/drained/e7.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 15; done
echo "$(date +%H:%M:%S) card 7 drained"; setst released
gpu stop selfedit-v4e-s7u092 | tail -1
gpu bg --name selfedit-v4e-w7 --gpus 7 "$(WCMD W7)" | tail -1
for n in W3 W7; do until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/w/$n.up.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 15; done; echo "$(date +%H:%M:%S) $n up"; done
echo "$(date +%H:%M:%S) to_w7 done"
