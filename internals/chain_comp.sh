#!/bin/bash
# fable_int Vesta, law table: one component through the whole measurement. chain_comp.sh NAME MODE [FIT_ITEMS TEST_ITEMS]
# NAME = folder work/c_NAME; MODE = intlib contrast mode (xhigh | low | sys@FILE | pre@FILE | proc with FI_BASE_NOPREFIX=1 | hind); base = "medium" (the prompt as read).
# Steps: attention cut + mixers + depth profile on the TEST reads; forcing capture on the FIT reads; rank-64 fit in all blocks; restoration R on the TEST reads by rank and band with a
# rotated-write and a random-read control; one summary line. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_c.sh; N=$1; MODE=$2; S=c_$N
FIT=${3:-$W/comp_fit_items.txt}; TEST=${4:-$W/comp_test_items.txt}; L=$W/$S/logs; mkdir -p $L
FI_MODE=$MODE $R $S a1_patch.py --rows $TEST --out a1 --sets cut,mix,res > $L/a1.log 2>&1
$R $S a3_inject.py --mode $MODE --rows $FIT --ns 384 --out a3/fit > $L/a3.log 2>&1
$R $S fit3p.py --a3 a3/fit --out cells --xs "" --rmax 64 --device cuda --val-fold 1 --base-arrays $C/arrays_keys_v2.npz --arrays-out arrays_$N.npz > $L/fit.log 2>&1
$R $S b_fn2.py --mode $MODE --bf16 --arrays cells/arrays_$N.npz --rows $TEST --out ev --cells "r8_all=0-63:8:1" "r64_all=0-63:64:1" "r64_full=0-63:64:1:blocks=full" "r64_lin=0-63:64:1:blocks=lin" "r8_L24-34=24-34:8:1" "r64_L24-34=24-34:64:1" "r64_L0-34=0-34:64:1" "r64_L35-63=35-63:64:1" "r64_all_wrot=0-63:64:1:var=wrot" "r64_all_rrot=0-63:64:1:var=rrot" "r8_L24-34_wrot=24-34:8:1:var=wrot" "r8_L24-34_rrot=24-34:8:1:var=rrot" > $L/ev.log 2>&1
$R $S comp_sum.py $N a1 ev > $W/$S/summary.txt 2>&1
echo "[chain_comp $N] done $(date)"
