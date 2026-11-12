#!/bin/bash
# fable_int Vesta, card A: HINDSIGHT, third launch. The first fit was stopped by me while it was saving its arrays (4 of 7 folds on disk): the fit is run again under the tag hh (5 minutes),
# then restoration R on held reads with the decoy-hint reference for the folds rows, babyvision, blink, charxiv. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
$R fit_lobo.py --a3 a3h/fit --out cells_h --tag hh --base-arrays $C/arrays_keys_v2.npz --rmax 64 > $L/fit_hh.log 2>&1
grep -q "arrays_hh_vlmsbiased.npz" $L/fit_hh.log && date > $L/fit_hh_done
CE='r8_all=0-63:8:1 r64_all=0-63:64:1 r64_full=0-63:64:1:blocks=full r64_all_wrot=0-63:64:1:var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for f in rows babyvision blink charxiv; do
  $R b_fn3.py --mode hind --alt --bf16 --arrays cells_h/arrays_hh_$f.npz --rows h_eval_${f}_items.txt --out ev_h3/$f --cells $CE > $L/ev_h3_$f.log 2>&1
done
date > $L/ev_h3_a_done
echo "[chain_h3_a] done $(date)"
