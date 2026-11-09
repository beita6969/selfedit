#!/bin/bash
# Mac side, 11-09: end-to-end dry run of 1109c-final: ONE real round in its own work dir p1109c_dry on 32 LEARN rows per bench (a subset of the 96),
# prereg_final with rounds 1, slots 6 (one floor slot per family), A1 size rule lowered (dry only) so a gate_v3 write (with its control) can be offered.
# Its results never count. Stock answers copied (cp -rL) from p1109c_full. Nothing deleted.
set -uo pipefail
SE=/workspace/fudan/mingdazhang/selfedit; N=$SE/res_driver_1109c_final2; H=/data/fd/mingdazhang/selfedit/rsi/mm/hb
W=$H/p1109c_dry; SRC=$H/p1109c_full; CL=$H/claims
TAG=$(python3 -c "import hashlib;print(hashlib.sha256('$W'.encode()).hexdigest()[:4])")
gpu run "test ! -e $W/state.json && mkdir -p $W/ev && cp -p $N/c1109deploy/prereg_final.json $W/prereg.json && cd $N && python3 - <<'PY'
import json, subprocess
a = json.load(open('$SRC/amendment.json')); a['exclude'] = '$SRC/exclude_96.json'
json.dump(a, open('$W/amendment96.json', 'w'), indent=1)
subprocess.run(['python3', 'c1109deploy/mk_sub.py', '$W/amendment96.json', '32', '$W/exclude_32.json'], check=True)
a['exclude'] = '$W/exclude_32.json'
a['rank'] = dict(a['rank'], n=22, n_note='dry: 89 x 32/128')
a['a1'] = dict(a['a1'], from_round=1, fit=2, read=1, score=1, note='DRY ONLY: size rule lowered so a gate write is offered on 32 rows')
json.dump(a, open('$W/amendment.json', 'w'), indent=1)
p = json.load(open('$W/prereg.json')); p.update(rounds=1, slots=6, written='DRY RUN of 1109c-final (1 round, 32 rows per bench, slots 6); never counts')
json.dump(p, open('$W/prereg.json', 'w'), indent=1)
PY
cp -rL $SRC/ev/aefc601d4c18b9 $W/ev/ && grep -o '\"a1\": {[^}]*}' $W/amendment.json | head -2; echo set up"
gpu run "grep -q ' hb${TAG}_' $CL/ALLOW || { cp -p $CL/ALLOW $CL/ALLOW.pre_hb${TAG} && echo ' hb${TAG}_' >> $CL/ALLOW; }; tail -2 $CL/ALLOW"
HEAD="SE=$SE; . \$SE/venv/bin/activate; export MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd $N; "
bash $(dirname "$0")/gbg.sh selfedit-hb_p1109c_dry_d --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py run $W"
echo "dry launched, tag hb$TAG"
