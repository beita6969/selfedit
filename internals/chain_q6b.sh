#!/bin/bash
# fable_int Vesta, card A queue after the hindsight evaluation: law-table components xhigh, low, PY note; then the rest of the hindsight patching reads and the opening-only hindsight fit.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh
until [ -f $L/ev_h5_a_done ]; do sleep 5; done
bash $C/chain_comp3.sh xhigh xhigh > $L/comp_xhigh.log 2>&1
bash $C/chain_comp3.sh low low > $L/comp_low.log 2>&1
bash $C/chain_comp3.sh pynote sys@$C/note_py.txt > $L/comp_pynote.log 2>&1
FI_MODE=hind $R a1_patch.py --rows h_cut_items.txt --out a1h --sets cut,mix,res > $L/a1h_4.log 2>&1
$R ana1.py a1h > $W/tables_h_a1.txt 2>&1
$R fit_lobo.py --a3 a3h/fit --out cells_h --tag hopen --base-arrays $C/arrays_keys_v2.npz --rmax 64 --tmax 150 --no-save > $L/fit_hopen.log 2>&1
echo "[chain_q6b] done $(date)"
