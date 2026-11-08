#!/bin/bash
# Mac side. Launch of the 128b run (prepared 11-08, NOT run). Order: deploy checked -> stock fill DONE -> claims prefixes -> driver + reporter. Card workers: cw/cw_launch.py (already running on the Mac) keeps cw<card>_<n> on cards 0-7 once open lines carry an ALLOW prefix.
# Nothing is deleted; ALLOW and LOW are backed up before the append.
set -euo pipefail
SE=/workspace/fudan/mingdazhang/selfedit
W=/data/fd/mingdazhang/selfedit/rsi/mm/hb/paper128b
CL=/data/fd/mingdazhang/selfedit/rsi/mm/hb/claims
GBG=${GBG:-/private/tmp/claude-501/-Users-admin/83609017-c4f2-4b7e-b0ee-236d61a3f0f6/scratchpad/gbg.sh}
# 0 checks: deploy applied (live shas = prereg code_shas), stock fill done, its workers gone
gpu run "cd $SE/code-xf && python3 -c \"
import json,hashlib
p=json.load(open('$W/prereg.json'))['code_shas']
bad=[k for k,v in p.items() if hashlib.sha256(open(k,'rb').read()).hexdigest()!=v]
print('live vs prereg code_shas: mismatched', bad); assert not bad
\" && test -e $W/stockfill_cards/DONE && echo stockfill DONE"
gpu jobs | grep -q "selfedit-sf128b-w.*running" && { echo "stock workers still running"; exit 1; }
# 1 card prefixes: run hb0b9d_, reporter hb8554_; reporter and side lines low
gpu run "cp -p $CL/ALLOW $CL/ALLOW.pre_p128b && cp -p $CL/LOW $CL/LOW.pre_p128b && echo ' hb0b9d_ hb8554_' >> $CL/ALLOW && echo ' hb8554_ hb0b9d_s' >> $CL/LOW && cat $CL/ALLOW $CL/LOW"
# 2 driver and reporter (CPU work; card 7 slot as paper1's)
HEAD="SE=$SE; . \$SE/venv/bin/activate; export MPLBACKEND=Agg PYTHONUNBUFFERED=1; cd \$SE/code-xf; "
bash $GBG selfedit-hb_p128b_d1 --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py run $W"
bash $GBG selfedit-hb_p128b_rl1 --gpus 7 --mem 2 "${HEAD}python -u rsi/hboost.py report $W --learn"
echo "launched; logs /data/fd/mingdazhang/gpu-logs/selfedit-hb_p128b_d1.log, run.txt in $W"
