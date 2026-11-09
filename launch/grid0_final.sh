#!/bin/bash
# Mac side, 11-09: the round-0 stock grid for 1109c-final with the fixed operators (op_patch keeps the stock token cap), run in its own work dir
# p1109c_grid0 (fresh copies of stock's answers, no symlinks), stopped with gpu stop once r001/grid.json exists. The final run reuses grid.json (disclosed).
set -uo pipefail
SE=/workspace/fudan/mingdazhang/selfedit; N=$SE/res_driver_1109c_final; H=/data/fd/mingdazhang/selfedit/rsi/mm/hb
W=$H/p1109c_grid0; SRC=$H/p1109c_full; CL=$H/claims
TAG=$(python3 -c "import hashlib;print(hashlib.sha256('$W'.encode()).hexdigest()[:4])")
gpu run "test ! -e $W/state.json && mkdir -p $W/ev && cp -p $N/c1109deploy/prereg_full.json $W/prereg.json && cp -p $SRC/exclude_96.json $W/ && python3 - <<'PY'
import json
a = json.load(open('$SRC/amendment.json')); a['exclude'] = '$W/exclude_96.json'
a['a1'] = dict(a['a1'], from_round=1, from_round_note='1109c-final audit 11-09: A1 gate-key eligibility from round 1 (r1 of p1109c_full wasted all 3 gate writes on ineligible keys)')
json.dump(a, open('$W/amendment.json', 'w'), indent=1)
p = json.load(open('$W/prereg.json')); p['rounds'] = 1; p['written'] = 'grid0 for 1109c-final: round 1 grid only (stopped after grid.json)'; json.dump(p, open('$W/prereg.json', 'w'), indent=1)
PY
cp -rL $SRC/ev/aefc601d4c18b9 $W/ev/ && echo set up"
gpu run "grep -q ' hb${TAG}_' $CL/ALLOW || { cp -p $CL/ALLOW $CL/ALLOW.pre_hb${TAG} && echo ' hb${TAG}_' >> $CL/ALLOW; }; cat $CL/ALLOW"
HEAD="SE=$SE; . \$SE/venv/bin/activate; export MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd $N; "
bash $(dirname "$0")/gbg.sh selfedit-hb_p1109c_grid0_d --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py run $W"
echo "grid0 launched, tag hb$TAG"
