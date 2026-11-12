#!/bin/bash
# fable_int Vesta, card A: HINDSIGHT (base prompt + one system line stating the row's gold answer): rows, forcing capture (shard 0), statistics, leave-one-benchmark-out fit,
# restoration R on held reads for the folds rows / babyvision / blink / charxiv. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int2; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
mkdir -p $L; TR=/data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json
until grep -q "^\[int\].*\(done\|FAILED\)" $L/h_rows.log 2>/dev/null; do sleep 3; done      # rows are made beforehand (CPU): run_x.sh mk_rows_h.py $TR h 16 8
$R a3_inject.py --mode hind --rows h_all_items.txt --ns 384 --out a3h/fit --shard 0/2 > $L/a3h_a.log 2>&1
until grep -q "^\[int\].*\(done\|FAILED\)" $L/a3h_b.log 2>/dev/null; do sleep 5; done
$R h_stats.py a3h/fit > $W/tables_h_stats.txt 2>&1
$R fit_lobo.py --a3 a3h/fit --out cells_h --tag h --base-arrays $C/arrays_keys_v2.npz --rmax 64 > $L/fit_h.log 2>&1
date > $L/fit_h_done
CE='r8_all=0-63:8:1 r64_all=0-63:64:1 r64_full=0-63:64:1:blocks=full r64_all_wrot=0-63:64:1:var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for f in rows babyvision blink charxiv; do
  $R b_fn2.py --mode hind --bf16 --arrays cells_h/arrays_h_$f.npz --rows h_eval_${f}_items.txt --out ev_h/$f --cells $CE > $L/ev_h_$f.log 2>&1
done
date > $L/ev_h_a_done
echo "[chain_h_a] done $(date)"
