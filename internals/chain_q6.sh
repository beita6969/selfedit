#!/bin/bash
# fable_int Vesta, card A queue after the hindsight evaluation: P2 step 1 (late image-attention boost, activation level), then law-table components xhigh, low, PY note.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs
until [ -f $L/ev_h5_a_done ]; do sleep 5; done
bash $C/chain_p2.sh > $L/chain_p2.log 2>&1
bash $C/chain_comp3.sh xhigh xhigh > $L/comp_xhigh.log 2>&1
bash $C/chain_comp3.sh low low > $L/comp_low.log 2>&1
bash $C/chain_comp3.sh pynote sys@$C/note_py.txt > $L/comp_pynote.log 2>&1
echo "[chain_q6] done $(date)"
