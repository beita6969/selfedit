#!/bin/bash
# Mac side, 11-09: amendment 9 at the r3/r4 boundary of p1109c_full (same pattern as boundary_a8.sh). Only c1109 a9/knob_ops/knob_min and the a9 block
# are added to the run's prereg (prereg.pre_a9.json kept); a8, carry state and batch_rows 50 stay. Nothing is deleted.
set -uo pipefail
SE=/workspace/fudan/mingdazhang/selfedit; N=$SE/res_driver_1109c_a5; W=/data/fd/mingdazhang/selfedit/rsi/mm/hb/p1109c_full
OLD=${OLD:-selfedit-hb_p1109c_full_d3}; NEW=${NEW:-selfedit-hb_p1109c_full_d4}; RN=${RN:-4}
until gpu run "grep -q 'r$RN: 1109c admitted W families' $W/run.txt && echo y" 2>/dev/null | grep -q '^y'; do sleep 30; done
echo "$(date '+%H:%M:%S') r$RN started; stopping $OLD"
gpu stop $OLD
gpu run "cd $W && cp -p prereg.json prereg.pre_a9.json && python3 - <<'PY'
import json
p = json.load(open('prereg.json')); a9 = json.load(open('$N/c1109deploy/amendment_a9.json'))
p['c1109'].update(a9=True, knob_ops=['tokens', 'verbosity'], knob_min=2); p['a9'] = a9
assert p['c1109']['a8'] is True and p['batch_rows'] == 50 and p['c1109']['a7'] is True
json.dump(p, open('prereg.json.new', 'w'), indent=1)
print('kept a7', p['c1109']['a7'], 'a8', p['c1109']['a8'], 'batch_rows', p['batch_rows'], '| added a9')
PY
mv -f prereg.json.new prereg.json && echo 'prereg a9 installed'"
HEAD="SE=$SE; . \$SE/venv/bin/activate; export MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd $N; "
bash $(dirname "$0")/gbg.sh $NEW --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py run $W"
echo "$(date '+%H:%M:%S') relaunched $NEW"
