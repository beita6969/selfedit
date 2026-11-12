#!/bin/bash
# after the first .92 rollout: image_limit 40 on cards 4-7, re-roll cards 1,2 at .92 (limit 40), then card 0 at .95 (test)
cd ~/selfedit/code-v4/engine_v4
until grep -q "rollout done" rollout_u92.log; do sleep 30; done
gpu run "cd /data/fd/mingdazhang/selfedit/v4 && python3 -c \"
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port'] in (8204,8205,8206,8207): x['image_limit']=40
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')
\""
echo "$(date +%H:%M:%S) image_limit 40 set on 4-7"
bash rollout2.sh 0.92 96 64 1 2
bash rollout2.sh 0.95 128 72 0
echo "$(date +%H:%M:%S) chain done"
