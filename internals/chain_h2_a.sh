#!/bin/bash
# fable_int Vesta, card A: HINDSIGHT, restoration R on held reads with the decoy-hint reference (b_fn3): folds rows, babyvision, blink, charxiv. Starts when the fit is done. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
until [ -f $L/fit_h_done ]; do sleep 5; done
CE='r8_all=0-63:8:1 r64_all=0-63:64:1 r64_full=0-63:64:1:blocks=full r64_all_wrot=0-63:64:1:var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for f in rows babyvision blink charxiv; do
  $R b_fn3.py --mode hind --alt --bf16 --arrays cells_h/arrays_h_$f.npz --rows h_eval_${f}_items.txt --out ev_h3/$f --cells $CE > $L/ev_h3_$f.log 2>&1
done
date > $L/ev_h3_a_done
echo "[chain_h2_a] done $(date)"
