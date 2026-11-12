#!/bin/bash
# fable_int Vesta, card B (gpu bg --gpus 7): P1: forcing capture (shard 1), then MLP-input captures for the route key: routed rows WITHOUT the procedure text, and unrouted rows. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
mkdir -p $L; export FI_BASE_NOPREFIX=1
until grep -q "^\[int\].*\(done\|FAILED\)" $L/rows.log 2>/dev/null; do sleep 3; done
$R a3_inject.py --mode proc --rows proc_fit2_items.txt --ns 512 --out a3p/fit --shard 1/2 > $L/a3_b.log 2>&1
cat $W/proc_fit_items.txt $W/proc_test_items.txt > $W/proc_routed_items.txt
cat $W/proc_unrfit_items.txt $W/proc_unrouted_items.txt > $W/proc_unall_items.txt
$R kcap.py --rows proc_routed_items.txt --out kcap_r > $L/kcap_r.log 2>&1
$R kcap.py --rows proc_unall_items.txt --out kcap_u > $L/kcap_u.log 2>&1
date > $L/kcap_done
echo "[chain_p1b] done $(date)"
