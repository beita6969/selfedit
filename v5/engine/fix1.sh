#!/bin/bash
# card 1 relaunched manually after name collision; mark up when ready, then (after card 0 chain done) re-roll card 2
cd ~/selfedit/code-v4/engine_v4
V4=/data/fd/mingdazhang/selfedit/v4
ts=$(date +%s)
until gpu run "curl -sf 127.0.0.1:8201/v1/models >/dev/null" >/dev/null 2>&1; do sleep 15
  gpu jobs | grep -q "selfedit-v4e-s1u092t0127 .*exited" && { echo "$(date +%H:%M:%S) card 1 FAILED again"; exit 1; }; done
gpu run "cd $V4 && python3 -c \"
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port']==8201: x.update(state='up',cap=64,max_num_seqs=96,gpu_util=0.92,job='selfedit-v4e-s1u092t0127',started=$ts,image_limit=40)
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')
\""
echo "$(date +%H:%M:%S) card 1: up at util 0.92, cap 64, limit 40"
until grep -q "chain done\|rollout done: 0\|FAILED to start" chain_after.log rollout_u92.log 2>/dev/null && grep -q "card 0: up at\|card 0: server FAILED" chain_after.log; do sleep 30; done
echo "$(date +%H:%M:%S) card 0 step finished; re-rolling card 2"
bash rollout2.sh 0.92 96 64 2
echo "$(date +%H:%M:%S) fix done"
