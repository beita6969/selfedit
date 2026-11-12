#!/bin/bash
# Generation 8 card plan (owner 01:35): E = cards 0,1; cards 2,3 -> W workers (W2, W3); cards 4,5 -> score workers; score re-dealt over 4 shards (c4..c7).
cd ~/selfedit/code-v4/engine_v4
V4=/data/fd/mingdazhang/selfedit/v4; O=$V4/signal; S=/workspace/fudan/mingdazhang/selfedit-signal
TR=$V4/tmp_coord_gset_t005r_traces.json; MD=/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B
PRE="unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=/data/fd/mingdazhang/selfedit/.hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/data/fd/mingdazhang/selfedit/.triton-sig; cd $O"
SCMD() { echo "$PRE; $S/venv/bin/python -u $S/score_worker.py --traces $TR --rows mixed --shard $1/4 --out $O/scores_t005r_v2 --model $MD"; }
WCMD() { echo "SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=\$D/.hf VLLM_CACHE_ROOT=\$D/.vllm-cache; cd \$D/v4 && python -u /workspace/fudan/mingdazhang/selfedit-v4/engine_v4/w_worker.py --name $1 --util 0.92 --max-num-seqs 96"; }
setst() { gpu run "cd $V4 && python3 -c \"
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port'] in (8202,8203,8204,8205): x['state']='$1'; x['note']='gen8: cards 2,3 W; 4,5 score'
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')
print('cards 2-5 ->','$1')\"" | tail -1; }
t0=$(gpu run "date +%s" | tail -1)
echo "$(date +%H:%M:%S) drain cards 2-5"; setst drain
for c in 2 3 4 5; do
  n=0; until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/q/drained/e$c.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 15; n=$((n+1)); [ $n -ge 60 ] && { echo "card $c drain timeout (15 min), releasing"; break; }; done
done
echo "$(date +%H:%M:%S) drained"; setst released
for j in selfedit-v4e-s2stt0622 selfedit-v4e-s3stt0605 selfedit-v4e-s4stt0622 selfedit-v4e-s5stt0622; do gpu stop $j | tail -1; done
gpu bg --name selfedit-v4e-w2g8 --gpus 2 "$(WCMD W2)" | tail -1
gpu bg --name selfedit-v4e-w3g8 --gpus 3 "$(WCMD W3)" | tail -1
# re-deal the score shards: stop c6/c7 (atomic per-trace writes), restart all four with i/4 (saved traces skipped)
gpu stop selfedit-v4s-c6c | tail -1; gpu stop selfedit-v4s-c7c | tail -1
i=0; for c in 4 5 6 7; do gpu bg --name selfedit-v4s-c${c}q --gpus $c "$(SCMD $i)" | tail -1; i=$((i+1)); done
for n in W2 W3; do m=0; until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/w/$n.up.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 15; m=$((m+1)); [ $m -ge 60 ] && { echo "$n not up after 15 min"; break; }; done; echo "$(date +%H:%M:%S) $n up"; done
echo "$(date +%H:%M:%S) gen8 done"
