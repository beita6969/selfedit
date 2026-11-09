#!/bin/bash
# Mac side, 11-09: amendment 8 at the r2/r3 boundary of p1109c_full. Waits for ROUND 2 and the r3 start (state.json saved before the ROUND line),
# stops the driver with gpu stop, adds the r2 candidates that meet the carry rule to carry_seed, installs prereg_full (+a8) as the run's prereg
# (old one kept as prereg.pre_a8.json), and resumes the driver from state.json. Nothing is deleted.
set -uo pipefail
SE=/workspace/fudan/mingdazhang/selfedit; N=$SE/res_driver_1109c_a5; W=/data/fd/mingdazhang/selfedit/rsi/mm/hb/p1109c_full
OLD=${OLD:-selfedit-hb_p1109c_full_d2}; NEW=${NEW:-selfedit-hb_p1109c_full_d3}
until gpu run "grep -q 'r3: 1109c admitted W families' $W/run.txt && echo y" 2>/dev/null | grep -q '^y'; do sleep 30; done
echo "$(date '+%H:%M:%S') r3 started; stopping $OLD"
gpu stop $OLD
gpu run "cd $W && cp -p prereg.json prereg.pre_a8.json && python3 - <<'PY'
import json
p = json.load(open('$N/c1109deploy/prereg_full.json'))
d = json.load(open('r002/c1109.json'))
q = [x['name'] for x in d['stage2'] if (x.get('a7') or {}).get('path') == 'member' and not x.get('admissible') and x['a7'].get('D', 0) > 0 and x['a7'].get('up', 0) > x['a7'].get('down', 0)]
seed = list(p['c1109']['carry_seed']) + sorted(q, key=lambda n: -next(x['a7']['D'] for x in d['stage2'] if x['name'] == n))
p['c1109']['carry_seed'] = seed[:p['c1109']['carry_cap']]
p['a8']['installed'] = 'r2/r3 boundary 11-09; carry_seed ' + str(p['c1109']['carry_seed']) + ' (r2 qualifiers ' + str(q) + ')'
old = json.load(open('prereg.pre_a8.json'))
diff = sorted(k for k in set(p) | set(old) if p.get(k) != old.get(k))
print('prereg keys changed:', diff, '| c1109 changed:', sorted(k for k in set(p['c1109']) | set(old['c1109']) if p['c1109'].get(k) != old['c1109'].get(k)))
json.dump(p, open('prereg.json.new', 'w'), indent=1)
PY
mv -f prereg.json.new prereg.json && echo 'prereg a8 installed'"
HEAD="SE=$SE; . \$SE/venv/bin/activate; export MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd $N; "
bash $(dirname "$0")/gbg.sh $NEW --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py run $W"
echo "$(date '+%H:%M:%S') relaunched $NEW"
