#!/bin/bash
cd ~/selfedit/code-v4/engine_v4
V4=/data/fd/mingdazhang/selfedit/v4
st() { gpu run "cd $V4 && python3 -c \"
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port']==8206: x['state']='$1'; x['cap']=x.get('cap',48) if '$1'=='drain' else 48
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')\""; echo "$(date +%H:%M:%S) card 6 -> $1"; }
t0=$(gpu run "date +%s" | tail -1)
st drain
until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/q/drained/e6.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 15; done
echo "$(date +%H:%M:%S) drained; serial test"
gpu run "cd $V4 && /workspace/fudan/mingdazhang/selfedit/venv/bin/python -u /workspace/fudan/mingdazhang/selfedit-v4/engine_v4/pc_serial.py 8206 2>&1 | grep -v Warning | tail -8"
st up
echo "$(date +%H:%M:%S) pc_serial done"
