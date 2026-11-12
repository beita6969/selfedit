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
for c in "$@"; do for i in $(seq 1 80); do n=$(gpu run "curl -s 127.0.0.1:820$c/metrics | grep -E '^vllm:num_requests_(running|waiting)' | awk '{s+=\$2} END{print s+0}'" 2>/dev/null | tr -d '\r' | tail -1); echo "$(date +%H:%M:%S) card $c in flight $n"; [ "$n" = "0" ] && break; sleep 15; done; done
for c in "$@"; do
  for j in $(gpu jobs | awk -v c=$c '$1 ~ "^selfedit-v4e-s"c && $2=="running" {print $1}'); do gpu stop $j | tail -1; done
  for i in $(seq 1 40); do u=$(gpu run "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c" 2>/dev/null | tr -d '\r' | tail -1); echo "$(date +%H:%M:%S) card $c memory used ${u} MiB"; [ -n "$u" ] && [ "$u" -lt 2000 ] 2>/dev/null && break; sleep 10; done
  gpu bg --name selfedit-v4e-s${c}ot$TAG --gpus $c "bash $ED/serve.sh 820$c 96 --gpu-memory-utilization 0.92 --chat-template $ED/chat_template_open_think.jinja" | tail -1
  echo "{\"820$c\": {\"role\": \"E\", \"state\": \"starting\", \"weights\": \"stock\", \"edit_sha\": null, \"model\": \"stock\", \"cap\": 40, \"max_num_seqs\": 96, \"gpu_util\": 0.92, \"image_limit\": 40, \"job\": \"selfedit-v4e-s${c}ot$TAG\", \"note\": \"open_think template e41e3664\"}}" > spec_st_$c.json
  gpu put spec_st_$c.json $V4/ckpt/spec_st_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_st_$c.json" | tail -1
done
left="$*"
while [ -n "$left" ]; do
  sleep 15; nl=""
  for c in $left; do
    if gpu run "curl -sf 127.0.0.1:820$c/v1/models >/dev/null" >/dev/null 2>&1; then
      echo "{\"820$c\": {\"state\": \"up\", \"started\": $(date +%s)}}" > spec_up_$c.json
      gpu put spec_up_$c.json $V4/ckpt/spec_up_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_up_$c.json" | tail -1
      echo "$(date +%H:%M:%S) card $c: stock E up (open_think template)"
    elif gpu jobs | grep -q "selfedit-v4e-s${c}ot$TAG .*exited"; then echo "$(date +%H:%M:%S) card $c: FAILED"
    else nl="$nl $c"; fi
  done
  left=$nl
done
echo "$(date +%H:%M:%S) to_stock done: $*"
