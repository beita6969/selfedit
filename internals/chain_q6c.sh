#!/bin/bash
# fable_int Vesta, card B after chain_h5_b has finished (it is doing the hindsight patching reads and the opening-only fit): law-table components xhigh, low, PY note. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs
until grep -q "chain_h5_b\] done" /data/fd/mingdazhang/gpu-logs/selfedit-int-h5b.log 2>/dev/null; do sleep 10; done
bash $C/chain_comp3.sh xhigh xhigh > $L/comp_xhigh.log 2>&1
bash $C/chain_comp3.sh low low > $L/comp_low.log 2>&1
bash $C/chain_comp3.sh pynote sys@$C/note_py.txt > $L/comp_pynote.log 2>&1
echo "[chain_q6c] done $(date)"
