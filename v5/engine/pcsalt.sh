#!/bin/bash
# B: at generation 6's propose/lab phase: deploy salted dispatcher (restart), cards 6 + 7 -> prefix caching on (roles E + PC), submit the A/A
cd ~/selfedit/code-v4/engine_v4
V4=/data/fd/mingdazhang/selfedit/v4; ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4
until gpu run "python3 -c \"
import json,sys
g=json.load(open('$V4/runs/main_1110/state.json'))['gens']; k=max(g,key=int); d=g[k].get('done') or []
sys.exit(0 if int(k)>=6 and 'parent' in d and 'route' not in d else 1)\"" >/dev/null 2>&1; do sleep 30; done
echo "$(date +%H:%M:%S) gen 6 propose/lab phase"
gpu put dispatch.py $ED/dispatch.py | tail -1
gpu stop selfedit-v4e-disp10 | tail -1
gpu bg --name selfedit-v4e-disp11 "SE=/workspace/fudan/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; export MPLBACKEND=Agg CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4; cd $V4 && taskset -c 84-95,180-191 python -u $ED/dispatch.py --stats-every 60" | tail -1
J6=$(gpu run "python3 -c \"import json;e=json.load(open('$V4/engines.json'));print([x['job'] for x in e['servers'] if x['port']==8206][0])\"" | tail -1)
J7=$(gpu run "python3 -c \"import json;e=json.load(open('$V4/engines.json'));print([x['job'] for x in e['servers'] if x['port']==8207][0])\"" | tail -1)
echo "stopping $J6 $J7"
JOBS_6=$J6 JOBS_7=$J7 bash pc_on.sh 6 7
gpu run "python3 $ED/mkpcsalt.py go"
echo "$(date +%H:%M:%S) pcsalt submitted"
