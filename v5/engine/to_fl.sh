#!/bin/bash
# usage: JOBS_<card>="job" to_fl.sh CARD...   (coordinator 11-11: FL +0.75 test endpoint on cards 6/7)
# release the card in engines.json (dispatcher drains it), stop its stock E job, start vLLM on the fl075 checkpoint
# (same serve.sh flags; served name "fl075"), register it as role T with the edit_sha of FLs8_L24-34_+0.75 (outside the stock E pool:
# the loop dispatcher only sends W/T reads of exactly that edit there), mark it up.
ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4; V4=/data/fd/mingdazhang/selfedit/v4
FL=/data/fd/mingdazhang/selfedit/v5/wckpt/fl075; ESHA=b68abd038657f195
cd ~/selfedit/code-v4/engine_v4
TAG=t$(date +%H%M)
for c in "$@"; do
  echo "{\"820$c\": {\"state\": \"released\"}}" > spec_rel_$c.json
  gpu put spec_rel_$c.json $V4/ckpt/spec_rel_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_rel_$c.json" | tail -1
done
sleep 8
for c in "$@"; do
  v="JOBS_$c"; for j in ${!v}; do gpu stop $j | tail -1; done
  gpu bg --name selfedit-v4e-f${c}fl$TAG --gpus $c "MODEL_DIR=$FL SERVED=fl075 bash $ED/serve.sh 820$c 96 --gpu-memory-utilization 0.92" | tail -1
  echo "{\"820$c\": {\"role\": \"T\", \"state\": \"starting\", \"weights\": \"fl075\", \"edit_sha\": \"$ESHA\", \"model\": \"fl075\", \"cap\": 32, \"max_num_seqs\": 96, \"gpu_util\": 0.92, \"image_limit\": 40, \"job\": \"selfedit-v4e-f${c}fl$TAG\", \"note\": \"FLs8_L24-34_+0.75 baked (v5/wckpt/fl075); test endpoint for the PY escalation client\"}}" > spec_fl_$c.json
  gpu put spec_fl_$c.json $V4/ckpt/spec_fl_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_fl_$c.json" | tail -1
done
left="$*"
while [ -n "$left" ]; do
  sleep 30; nl=""
  for c in $left; do
    if gpu run "curl -sf 127.0.0.1:820$c/v1/models | grep -q fl075" >/dev/null 2>&1; then
      echo "{\"820$c\": {\"state\": \"up\", \"started\": $(date +%s)}}" > spec_up_$c.json
      gpu put spec_up_$c.json $V4/ckpt/spec_up_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_up_$c.json" | tail -1
      echo "$(date +%H:%M:%S) card $c: fl075 T up"
    elif gpu jobs | grep -q "selfedit-v4e-f${c}fl$TAG .*exited"; then echo "$(date +%H:%M:%S) card $c: FAILED"
    else nl="$nl $c"; fi
  done
  left=$nl
done
echo "$(date +%H:%M:%S) to_fl done: $*"
