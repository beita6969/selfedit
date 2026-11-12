#!/bin/bash
# fable_int Vesta, card B: HINDSIGHT: forcing capture (shard 1), attention cut + mixers + depth profile on 18 reads, restoration R for the folds lego / visulogic / vlmsbiased,
# then the opening-only fit (tail + first 150 thinking tokens). New files only.
C=/workspace/fudan/mingdazhang/selfedit-int2; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
mkdir -p $L
until grep -q "^\[int\].*\(done\|FAILED\)" $L/h_rows.log 2>/dev/null; do sleep 3; done
$R a3_inject.py --mode hind --rows h_all_items.txt --ns 384 --out a3h/fit --shard 1/2 > $L/a3h_b.log 2>&1
FI_MODE=hind $R a1_patch.py --rows h_cut_items.txt --out a1h --sets cut,mix,res > $L/a1h.log 2>&1
$R ana1.py a1h > $W/tables_h_a1.txt 2>&1
until [ -f $L/fit_h_done ]; do sleep 5; done
CE='r8_all=0-63:8:1 r64_all=0-63:64:1 r64_full=0-63:64:1:blocks=full r64_all_wrot=0-63:64:1:var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for f in lego visulogic vlmsbiased; do
  $R b_fn2.py --mode hind --bf16 --arrays cells_h/arrays_h_$f.npz --rows h_eval_${f}_items.txt --out ev_h/$f --cells $CE > $L/ev_h_$f.log 2>&1
done
until [ -f $L/ev_h_a_done ]; do sleep 5; done
$R ana_h.py ev_h/rows ev_h/babyvision ev_h/blink ev_h/charxiv ev_h/lego ev_h/visulogic ev_h/vlmsbiased > $W/tables_h_R.txt 2>&1
$R fit_lobo.py --a3 a3h/fit --out cells_h --tag hopen --base-arrays $C/arrays_keys_v2.npz --rmax 64 --tmax 150 --no-save > $L/fit_hopen.log 2>&1
echo "[chain_h_b] done $(date)"
