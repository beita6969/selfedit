#!/bin/bash
# usage: to_vp.sh CARD:ARM:OLDJOB ...   ARM in vp2|vpm|vp11 (coordinator 11-12 vision front-end track)
# release the card, stop its old job, serve the edited-vision checkpoint with serve.sh (same flags; served name = ARM), register it as role
# ARM upper-case (dispatcher role outside the stock pool: only requests with engine "VP2"/"VPM"/"VP11" go there), mark it up.
ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4; V4=/data/fd/mingdazhang/selfedit/v4
dir_of() { case $1 in vp2) echo /data/fd/mingdazhang/selfedit/models/vh/8fab893f70b1;; vpm) echo /data/fd/mingdazhang/selfedit/v5/wckpt/vpm;; vp11) echo /data/fd/mingdazhang/selfedit/v5/wckpt/vp11;; w2) echo /data/fd/mingdazhang/selfedit/v5/wckpt/w2_vp2m512_PRUa64;; w3) echo /data/fd/mingdazhang/selfedit/v5/wckpt/w3;; vp2m512) echo /data/fd/mingdazhang/selfedit/v5/wckpt/vp2m512;; vp2m) echo /data/fd/mingdazhang/selfedit/v5/wckpt/vp2m;; vp4m) echo /data/fd/mingdazhang/selfedit/models/vh/249e5adffb52;; *) echo BAD; exit 1;; esac; }
cd ~/selfedit/code-v4/engine_v4
TAG=t$(date +%H%M); cards=""
for spec in "$@"; do
  IFS=: read c arm old <<< "$spec"; cards="$cards $c"
  echo "{\"820$c\": {\"state\": \"released\"}}" > spec_rel_$c.json
  gpu put spec_rel_$c.json $V4/ckpt/spec_rel_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_rel_$c.json" | tail -1
done
sleep 8
for spec in "$@"; do
  IFS=: read c arm old <<< "$spec"; R=$(echo $arm | tr a-z A-Z)
  [ -n "$old" ] && gpu stop $old | tail -1
  gpu bg --name selfedit-v4e-v${c}${arm}$TAG --gpus $c "MODEL_DIR=$(dir_of $arm) SERVED=$arm bash $ED/serve.sh 820$c 96 --gpu-memory-utilization 0.92" | tail -1
  echo "{\"820$c\": {\"role\": \"$R\", \"state\": \"starting\", \"weights\": \"$arm\", \"edit_sha\": null, \"model\": \"$arm\", \"cap\": 40, \"max_num_seqs\": 96, \"gpu_util\": 0.92, \"image_limit\": 40, \"job\": \"selfedit-v4e-v${c}${arm}$TAG\", \"note\": \"edited vision front-end $(dir_of $arm)\"}}" > spec_vp_$c.json
  gpu put spec_vp_$c.json $V4/ckpt/spec_vp_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_vp_$c.json" | tail -1
done
left="$cards"
while [ -n "$left" ]; do
  sleep 30; nl=""
  for c in $left; do
    arm=$(for spec in "$@"; do IFS=: read cc aa oo <<< "$spec"; [ "$cc" = "$c" ] && echo $aa; done)
    if gpu run "curl -sf 127.0.0.1:820$c/v1/models | grep -q '\"$arm\"'" >/dev/null 2>&1; then
      echo "{\"820$c\": {\"state\": \"up\", \"started\": $(date +%s)}}" > spec_up_$c.json
      gpu put spec_up_$c.json $V4/ckpt/spec_up_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_up_$c.json" | tail -1
      echo "$(date +%H:%M:%S) card $c: $arm up"
    elif gpu jobs | grep -q "selfedit-v4e-v${c}${arm}$TAG .*exited"; then echo "$(date +%H:%M:%S) card $c: $arm FAILED"
    else nl="$nl $c"; fi
  done
  left=$nl
done
echo "$(date +%H:%M:%S) to_vp done:$cards"
