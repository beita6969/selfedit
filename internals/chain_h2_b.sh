#!/bin/bash
# fable_int Vesta, card B: HINDSIGHT, restoration R with the decoy-hint reference: folds lego, visulogic, vlmsbiased; then the pooled table and the opening-only fit. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
until [ -f $L/fit_h_done ]; do sleep 5; done
CE='r8_all=0-63:8:1 r64_all=0-63:64:1 r64_full=0-63:64:1:blocks=full r64_all_wrot=0-63:64:1:var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for f in lego visulogic vlmsbiased; do
  $R b_fn3.py --mode hind --alt --bf16 --arrays cells_h/arrays_h_$f.npz --rows h_eval_${f}_items.txt --out ev_h3/$f --cells $CE > $L/ev_h3_$f.log 2>&1
done
until [ -f $L/ev_h3_a_done ]; do sleep 5; done
$R ana_h2.py ev_h3/rows ev_h3/babyvision ev_h3/blink ev_h3/charxiv ev_h3/lego ev_h3/visulogic ev_h3/vlmsbiased > $W/tables_h_R.txt 2>&1
$R fit_lobo.py --a3 a3h/fit --out cells_h --tag hopen --base-arrays $C/arrays_keys_v2.npz --rmax 64 --tmax 150 --no-save > $L/fit_hopen.log 2>&1
echo "[chain_h2_b] done $(date)"
