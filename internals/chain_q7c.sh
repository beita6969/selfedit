#!/bin/bash
# fable_int Vesta, card B: the rest of P2 step 1 (reads 7-10 with the chunked tables; reads 1-6 are on disk and are skipped), then the law-table components xhigh, low, PY note.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh
until grep -q "chain_h5_b\] done" /data/fd/mingdazhang/gpu-logs/selfedit-int-h5b.log 2>/dev/null; do sleep 10; done
$R p2_reopen2.py --out p2a --n 10 --ton 1000 --gains 1.0,2.0 --ps-gain 2.0 > $L/p2a_2.log 2>&1
grep -v -i "warn\|falling" $L/p2a_2.log | sed -n '/reads (/,$p' > $W/tables_p2a.txt
$R ana_p2.py p2a 1000 > $W/tables_p2a_new.txt 2>&1
bash $C/chain_comp3.sh xhigh xhigh > $L/comp_xhigh.log 2>&1
bash $C/chain_comp3.sh low low > $L/comp_low.log 2>&1
bash $C/chain_comp3.sh pynote sys@$C/note_py.txt > $L/comp_pynote.log 2>&1
echo "[chain_q7c] done $(date)"
