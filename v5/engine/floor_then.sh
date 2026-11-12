#!/bin/bash
cd ~/selfedit/code-v4/engine_v4
V4=/data/fd/mingdazhang/selfedit/v4; ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4
until grep -q "switch done" to_t.log; do sleep 10; done
gpu run "cd $V4 && python3 -c \"
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port']==8204: x['state']='released'; x['note']='stock floor test'
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')\""
echo "$(date +%H:%M:%S) card 4 released; floor job on card 7"
gpu bg --name selfedit-v4e-floor --gpus 7 "SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=\$D/.hf VLLM_CACHE_ROOT=\$D/.vllm-cache; cd \$D/v4 && python -u $ED/floor.py 8204" | tail -1
until gpu run "grep -q 'stock floor:' /data/fd/mingdazhang/gpu-logs/selfedit-v4e-floor.log" >/dev/null 2>&1; do sleep 15; gpu jobs | grep -q "selfedit-v4e-floor .*exited(" && break; done
gpu run "grep -E 'row|floor' /data/fd/mingdazhang/gpu-logs/selfedit-v4e-floor.log"
echo "$(date +%H:%M:%S) floor done"
