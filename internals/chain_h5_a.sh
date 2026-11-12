#!/bin/bash
# fable_int Vesta, card A: HINDSIGHT evaluation, fifth launch (b_fn5: per-position statistics in chunks; finished reads are kept): folds babyvision, charxiv, rows.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
CE='r8_all=0-63:8:1 r64_all=0-63:64:1 r64_full=0-63:64:1:blocks=full r64_all_wrot=0-63:64:1:var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for f in babyvision rows; do
  $R b_fn5.py --mode hind --alt --bf16 --arrays cells_h/arrays_hh_$f.npz --rows h_eval_${f}_items.txt --out ev_h3/$f --cells $CE > $L/ev_h5_$f.log 2>&1
done
date > $L/ev_h5_a_done
echo "[chain_h5_a] done $(date)"
