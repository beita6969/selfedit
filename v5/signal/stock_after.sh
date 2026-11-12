#!/bin/bash
# When a short-set score job ends, turn its card into a stock E server (coordinator 03:40).
pairs="4:selfedit-v4s-f32c4e 5:selfedit-v4s-f32c5e 6:selfedit-v4s-f32c6 7:selfedit-v4s-f32c7e2"
left="$pairs"
while [ -n "$left" ]; do
  J=$(gpu jobs 2>/dev/null); nl=""
  [ -z "$J" ] && { sleep 60; continue; }
  for p in $left; do
    c=${p%%:*}; j=${p#*:}
    st=$(echo "$J" | awk -v n="$j" '$1==n{print $2}')
    if [ -n "$st" ] && [ "$st" != running ]; then
      echo "$(date +%H:%M:%S) $j $st -> card $c to stock"
      ( cd ~/selfedit/code-v4/engine_v4 && bash to_stock.sh $c > to_stock_c$c.log 2>&1; tail -1 to_stock_c$c.log ) &
    else nl="$nl $p"; fi
  done
  left=$nl
  sleep 60
done
wait
