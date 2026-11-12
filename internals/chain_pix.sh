#!/bin/bash
# fable_int Vesta, card C after the E1 chain: the rest of P2 step 1 (reads 7-10), then the law-table row C_pix: cached t005r reads teacher-forced under the stock lattice against the VP2
# lattice (patch 8), both on the original image; forcing capture, where it lives, leave-one-benchmark-out explained energy with controls. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh
until grep -q "chain_p1u\] done" /data/fd/mingdazhang/gpu-logs/selfedit-int-p1u.log 2>/dev/null; do sleep 10; done
$R p2_reopen2.py --out p2a --n 10 --ton 1000 --gains 1.0,2.0 --ps-gain 2.0 > $L/p2a_2.log 2>&1
grep -v -i "warn\|falling" $L/p2a_2.log | sed -n '/reads (/,$p' > $W/tables_p2a_10.txt
$R ana_p2.py p2a 1000 > $W/tables_p2a_new.txt 2>&1
mkdir -p $W/c_pix/logs
$RC c_pix mk_rows_pix.py /data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json pix_items.txt 10 > $W/c_pix/logs/rows.log 2>&1
$RC c_pix pix_run.py a3_inject2.py --mode vp2 --no-att --rows pix_items.txt --ns 192 --out a3/fit > $W/c_pix/logs/a3.log 2>&1
$RC c_pix h_stats.py a3/fit > $W/c_pix/tables_pix_stats.txt 2>&1
$RC c_pix fit_lobo2.py --a3 a3/fit --out cells --tag pix --base-arrays $C/arrays_keys_v2.npz --rmax 64 --no-save > $W/c_pix/logs/fit.log 2>&1
$RC c_pix ana_pix.py a3/fit cells/ev_pix.npz > $W/c_pix/summary.txt 2>&1
echo "[chain_pix] done $(date)"
