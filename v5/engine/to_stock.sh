#!/bin/bash
# usage: to_stock.sh CARD...   stop whatever selfedit-v4e job runs on each card (T server / W worker / cell server: give the job names
# via JOBS_<card> env or they are looked up from engines.json "job"), start a stock E server (util .92, max_num_seqs 96), mark it up (cap 40).
ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4; V4=/data/fd/mingdazhang/selfedit/v4
cd ~/selfedit/code-v4/engine_v4
TAG=t$(date +%H%M)
for c in "$@"; do
  echo "{\"820$c\": {\"state\": \"released\"}}" > spec_rel_$c.json
  gpu put spec_rel_$c.json $V4/ckpt/spec_rel_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_rel_$c.json" | tail -1
done
sleep 8
for c in "$@"; do
  v="JOBS_$c"; jobs="${!v}"
  for j in $jobs; do gpu stop $j | tail -1; done
  gpu bg --name selfedit-v4e-s${c}st$TAG --gpus $c "bash $ED/serve.sh 820$c 96 --gpu-memory-utilization 0.92" | tail -1
  echo "{\"820$c\": {\"role\": \"E\", \"state\": \"starting\", \"weights\": \"stock\", \"edit_sha\": null, \"model\": \"stock\", \"cap\": 40, \"max_num_seqs\": 96, \"gpu_util\": 0.92, \"image_limit\": 40, \"job\": \"selfedit-v4e-s${c}st$TAG\", \"note\": null}}" > spec_st_$c.json
  gpu put spec_st_$c.json $V4/ckpt/spec_st_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_st_$c.json" | tail -1
done
left="$*"
while [ -n "$left" ]; do
  sleep 30; nl=""
  for c in $left; do
    if gpu run "curl -sf 127.0.0.1:820$c/v1/models >/dev/null" >/dev/null 2>&1; then
      echo "{\"820$c\": {\"state\": \"up\", \"started\": $(date +%s)}}" > spec_up_$c.json
      gpu put spec_up_$c.json $V4/ckpt/spec_up_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_up_$c.json" | tail -1
      echo "$(date +%H:%M:%S) card $c: stock E up"
    elif gpu jobs | grep -q "selfedit-v4e-s${c}st$TAG .*exited"; then echo "$(date +%H:%M:%S) card $c: FAILED"
    else nl="$nl $c"; fi
  done
  left=$nl
done
echo "$(date +%H:%M:%S) to_stock done: $*"
