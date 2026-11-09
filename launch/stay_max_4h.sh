#!/bin/bash
# Coordinator 11-09 ~01:35 CST: between round 1 and round 2 of p1109c_final_c, raise the card code's resident-chaining cap STAY_MAX 3600 -> 14400 s
# in code-xf/rsi/hboost.py (card-side job-batch; not a git tree: backup copy + shas logged). Scheduling only: chain_batches takes a line only if its
# command tail (checkpoint, settings, --seed) equals the loaded engine's; completion is read from the line's outs / .rc, which a chained line writes
# itself; driver patience runs from the wait start; copy_stranded looks at claim age only when the Mac mark is stale. No line needs STAY_MAX for correctness.
set -euo pipefail
F=/workspace/fudan/mingdazhang/selfedit/code-xf/rsi/hboost.py; W=/data/fd/mingdazhang/selfedit/rsi/mm/hb/p1109c_final_c
until gpu run "python3 -c \"import json;print('R', json.load(open('$W/state.json')).get('round'))\"" 2>/dev/null | grep -qE '^R [1-9]'; do sleep 60; done
gpu run "grep -q '^STAY_IDLE, STAY_MAX = 45, 3600$' $F && cp -p $F $F.pre_stay4h && sed -i 's/^STAY_IDLE, STAY_MAX = 45, 3600$/STAY_IDLE, STAY_MAX = 45, 14400      # coordinator 11-09: 4 h engine reuse (scheduling only), between r1 and r2 of p1109c_final_c/' $F && python3 -c \"import ast;ast.parse(open('$F').read())\" && echo \"\$(date '+%F %T %Z') STAY_MAX 3600 -> 14400 in code-xf/rsi/hboost.py (card code), between r1 and r2; sha before \$(sha256sum < $F.pre_stay4h | cut -c1-16) after \$(sha256sum < $F | cut -c1-16); commit c7dab7e\" | tee -a $W/run_notes.txt; grep -n '^STAY_IDLE' $F"
