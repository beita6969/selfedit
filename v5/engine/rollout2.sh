#!/bin/bash
# Mac side. Rolling restart of E servers to a new gpu util, one at a time: drain -> stop -> start -> up (cap CAP).
# usage: bash rollout.sh UTIL MAXSEQS CAP CARD...      e.g. bash rollout.sh 0.92 96 64 1 2 4 5 6 7
UTIL=$1; MNS=$2; CAP=$3; shift 3
V4=/data/fd/mingdazhang/selfedit/v4
TAG=u$(echo $UTIL | tr -d .)t$(date +%H%M)
setstate() {  # card state [extra json]
  local EX='{}'; [ -n "$3" ] && EX="$3"
  gpu run "cd $V4 && python3 - <<EOF
import json,time,os
e=json.load(open('engines.json'))
for x in e['servers']:
    if x['port']==8200+$1:
        x['state']='$2'; x.update(json.loads('''$EX'''))
e['updated']=time.time(); json.dump(e,open('engines.json.tmp','w'),indent=1); os.replace('engines.json.tmp','engines.json')
print('card',$1,'->','$2')
EOF" | tail -1
}
for c in "$@"; do
  t0=$(gpu run "date +%s" | tail -1)
  echo "$(date +%H:%M:%S) card $c: drain"
  setstate $c drain
  until gpu run "python3 -c \"import json,sys; d=json.load(open('$V4/q/drained/e$c.json')); sys.exit(0 if d['t']>$t0 else 1)\"" >/dev/null 2>&1; do sleep 20; done
  echo "$(date +%H:%M:%S) card $c: drained, restarting at util $UTIL"
  old=$(gpu jobs | awk -v c=$c '$1 ~ "^selfedit-v4e-s"c"(b|u[0-9]+(t[0-9]+)?)?$" && $2=="running" {print $1}')
  for j in $old; do gpu stop $j | tail -1; done
  gpu bg --name selfedit-v4e-s$c$TAG --gpus $c "bash /workspace/fudan/mingdazhang/selfedit-v4/engine_v4/serve.sh 820$c $MNS --gpu-memory-utilization $UTIL $EXTRA" | tail -1
  ts=$(date +%s)
  until gpu run "curl -sf 127.0.0.1:820$c/v1/models >/dev/null" >/dev/null 2>&1; do
    sleep 15
    if gpu jobs | grep -q "selfedit-v4e-s$c$TAG .*exited"; then echo "$(date +%H:%M:%S) card $c: server FAILED to start"; exit 1; fi
  done
  setstate $c up "{\"cap\": $CAP, \"max_num_seqs\": $MNS, \"gpu_util\": $UTIL, \"job\": \"selfedit-v4e-s$c$TAG\", \"started\": $ts, \"image_limit\": 40}"
  echo "$(date +%H:%M:%S) card $c: up at util $UTIL, cap $CAP ($(( $(date +%s) - ts )) s start)"
done
echo "$(date +%H:%M:%S) rollout done: $*"
