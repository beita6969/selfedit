#!/bin/bash
# fable_int Vesta, card A after chain_q7b (ao1, g005c29, g005c23b): the law-table row C_pix (cached t005r reads teacher-forced under the stock lattice against the VP2 lattice, both on the
# original image; 60 single-image reads; leave-one-benchmark-out explained forcing energy with controls). Own file names (suffix c): card C holds the same steps behind E1b. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; RC=$C/run_c.sh
until grep -q "chain_q7b\] done" /data/fd/mingdazhang/gpu-logs/selfedit-int-q7b.log 2>/dev/null; do sleep 10; done
$RC c_pix pix_run.py a3_inject2.py --mode vp2 --no-att --rows pix_items.txt --ns 192 --out a3c/fit > $W/c_pix/logs/a3c.log 2>&1
$RC c_pix h_stats.py a3c/fit > $W/c_pix/tables_pix_stats_c.txt 2>&1
$RC c_pix fit_lobo2.py --a3 a3c/fit --out cells_c --tag pix --base-arrays $C/arrays_keys_v2.npz --rmax 64 --no-save > $W/c_pix/logs/fit_c.log 2>&1
$RC c_pix ana_pix.py a3c/fit cells_c/ev_pix.npz > $W/c_pix/summary_c.txt 2>&1
echo "[chain_q6e] done $(date)"
