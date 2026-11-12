#!/bin/bash
# usage: mark_up.sh CARD...   mark T servers "up" in engines.json as their /v1/models answers
ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4; V4=/data/fd/mingdazhang/selfedit/v4
left="$*"
while [ -n "$left" ]; do
  sleep 15; nl=""
  for c in $left; do
    if gpu run "curl -sf 127.0.0.1:820$c/v1/models >/dev/null" >/dev/null 2>&1; then
      echo "{\"820$c\": {\"state\": \"up\", \"started\": $(date +%s)}}" > /tmp/spec_up_$c.json 2>/dev/null || echo "{\"820$c\": {\"state\": \"up\"}}" > spec_up_$c.json
      cp /tmp/spec_up_$c.json spec_up_$c.json 2>/dev/null
      gpu put spec_up_$c.json $V4/ckpt/spec_up_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_up_$c.json" | tail -1
      echo "$(date +%H:%M:%S) card $c: up"
    else nl="$nl $c"; fi
  done
  left=$nl
done
echo "$(date +%H:%M:%S) mark_up done"
