#!/bin/bash
# Mac side, 11-09: launch of 1109c-final (fresh dir p1109c_final, stock only, 96 rows per bench, 20 rounds, prereg_final.json on the
# res_driver_1109c_final2 tree; code shas checked). Stock answers and the round-1 grid cell answers are COPIED (cp -rL, no symlinks) from p1109c_full
# (stock) and p1109c_grid0 (grid cells and stock redraws on the same 96 rows with the fixed ops): the round-1 grid is rebuilt from that cache (disclosed).
set -euo pipefail
SE=/workspace/fudan/mingdazhang/selfedit; N=$SE/res_driver_1109c_final2; H=/data/fd/mingdazhang/selfedit/rsi/mm/hb
W=$H/p1109c_final_c; SRC=$H/p1109c_full; G0=$H/p1109c_grid0; CL=$H/claims
TAG=$(python3 -c "import hashlib;print(hashlib.sha256('$W'.encode()).hexdigest()[:4])")
gpu run "cd $N && python3 -c \"
import json,hashlib
p=json.load(open('c1109deploy/prereg_final.json'))['code_shas_1109c']
bad=[k for k,v in p.items() if hashlib.sha256(open(k,'rb').read()).hexdigest()!=v]
print('final2 tree vs prereg_final code shas: mismatched', bad); assert not bad
\""
gpu run "test ! -e $W/state.json && mkdir -p $W/ev && cp -p $N/c1109deploy/prereg_final.json $W/prereg.json && cp -p $SRC/exclude_96.json $W/ && python3 - <<'PY'
import json
a = json.load(open('$SRC/amendment.json')); a['exclude'] = '$W/exclude_96.json'
a['a1'] = dict(a['a1'], from_round=1, from_round_note='1109c-final audit 11-09: A1 gate-key eligibility from round 1')
a['grid_reuse'] = 'round-1 grid cell answers and stock redraws copied from p1109c_grid0 (same tree code, same 96 rows, fixed ops, stock only); stock answers copied from p1109c_full'
json.dump(a, open('$W/amendment.json', 'w'), indent=1)
PY
cp -rL $SRC/ev/aefc601d4c18b9 $W/ev/ && for e in \$(ls $G0/ev); do [ -e $W/ev/\$e ] || cp -rL $G0/ev/\$e $W/ev/; done; ls $W/ev | wc -l; echo set up"
gpu run "grep -q ' hb${TAG}_' $CL/ALLOW || { cp -p $CL/ALLOW $CL/ALLOW.pre_hb${TAG} && cp -p $CL/LOW $CL/LOW.pre_hb${TAG} && echo ' hb${TAG}_' >> $CL/ALLOW && echo ' hb${TAG}_s' >> $CL/LOW; }; tail -1 $CL/ALLOW; tail -1 $CL/LOW"
# card-side code: eval jobs (hboost.py job-batch) run in the card tree code-xf by design (bl_worker cd $SE/code-xf; same code as stock + grid0 answers); gate writes cd into final2
gpu run "cd $SE/code-xf && sha256sum rsi/hboost.py rsi/mm_eval.py rsi/vision_patch.py rsi/vharness.py mm/gen_mm.py mm/grade_mm.py > $W/card_code_shas.txt; cat $W/card_code_shas.txt | cut -c1-16,65-"
HEAD="SE=$SE; . \$SE/venv/bin/activate; export MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd $N; "
bash $(dirname "$0")/gbg.sh selfedit-hb_p1109c_finalc_d --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py run $W"
bash $(dirname "$0")/gbg.sh selfedit-hb_p1109c_finalc_rl --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py report $W --learn"
echo "final launched, tag hb$TAG; run.txt in $W"
