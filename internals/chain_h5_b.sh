#!/bin/bash
# fable_int Vesta, card B: HINDSIGHT evaluation, fifth launch (b_fn5): folds visulogic, vlmsbiased (blink and lego are done); pooled table; the remaining patching reads; opening-only fit.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
CE='r8_all=0-63:8:1 r64_all=0-63:64:1 r64_full=0-63:64:1:blocks=full r64_all_wrot=0-63:64:1:var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for f in visulogic vlmsbiased; do
  $R b_fn5.py --mode hind --alt --bf16 --arrays cells_h/arrays_hh_$f.npz --rows h_eval_${f}_items.txt --out ev_h3/$f --cells $CE > $L/ev_h5_$f.log 2>&1
done
$R ana_h2.py ev_h3/blink ev_h3/lego ev_h3/visulogic ev_h3/vlmsbiased > $W/tables_h_R_b2.txt 2>&1
until [ -f $L/ev_h5_a_done ]; do sleep 5; done
$R ana_h2.py ev_h3/rows ev_h3/babyvision ev_h3/blink ev_h3/charxiv ev_h3/lego ev_h3/visulogic ev_h3/vlmsbiased > $W/tables_h_R.txt 2>&1
FI_MODE=hind $R a1_patch.py --rows h_cut_items.txt --out a1h --sets cut,mix,res > $L/a1h_3.log 2>&1
$R ana1.py a1h > $W/tables_h_a1.txt 2>&1
$R fit_lobo.py --a3 a3h/fit --out cells_h --tag hopen --base-arrays $C/arrays_keys_v2.npz --rmax 64 --tmax 150 --no-save > $L/fit_hopen.log 2>&1
echo "[chain_h5_b] done $(date)"
