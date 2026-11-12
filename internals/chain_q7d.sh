#!/bin/bash
# fable_int Vesta, card B after chain_q6c (xhigh, low, PY note): law-table measurements on reads generated WITHOUT the routed text. (1) g002c26: attention cut, mixers and depth profile on
# the text-free reads and on the text-generated reads of the held-out routed rows. (2) g005c29 as a whole component on text-free reads (floor-only reads of its rows). New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces
FT=$T/floor_signal_routed.json; EX=/data/fd/mingdazhang/selfedit/v4/scratch/v5_exports/e13_floor_signal_full_s0-7.jsonl.gz
until grep -q "chain_q6c\] done" /data/fd/mingdazhang/gpu-logs/selfedit-int-q6c.log 2>/dev/null; do sleep 10; done
export FI_BASE_NOPREFIX=1 FI_MAX_REASON=1500
mkdir -p $W/c_g002c26x/logs $W/c_g002c26t/logs $W/c_g005c29n/logs
FI_TRACES=$FT FI_MODE=proc $RC c_g002c26x a1_patch.py --rows $W/proc_test_floor_items.txt --shard 0/2 --out a1 --sets cut,mix,res > $W/c_g002c26x/logs/a1.log 2>&1
FI_TRACES=$FT $RC c_g002c26x ana1.py a1 > $W/c_g002c26x/tables_a1.txt 2>&1
FI_MODE=proc $RC c_g002c26t a1_patch.py --rows $W/proc_test_items.txt --out a1 --sets cut,mix,res > $W/c_g002c26t/logs/a1.log 2>&1
$RC c_g002c26t ana1.py a1 > $W/c_g002c26t/tables_a1.txt 2>&1
$RC c_g005c29n conv_floor_parts.py $T/t007c_g005c29.json $EX $T/floor_g005c29.json > $W/c_g005c29n/logs/rows.log 2>&1
$RC c_g005c29n mk_items4.py $T/floor_g005c29.json $W/c_g005c29/proc_fit_items.txt proc_fit_items.txt 1 150 >> $W/c_g005c29n/logs/rows.log 2>&1
$RC c_g005c29n mk_items4.py $T/floor_g005c29.json $W/c_g005c29/proc_test_items.txt proc_test_items.txt 1 150 >> $W/c_g005c29n/logs/rows.log 2>&1
FI_TRACES=$T/floor_g005c29.json bash $C/chain_comp3.sh g005c29n proc $W/c_g005c29n/proc_fit_items.txt $W/c_g005c29n/proc_test_items.txt > $L/comp_g005c29n.log 2>&1
echo "[chain_q7d] done $(date)"
