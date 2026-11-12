#!/bin/bash
# fable_int Vesta: P2 step 1 (activation level): late boost of the stream's attention to the image tokens on cached long reads, real against blank image. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
$R p2_reopen.py --out p2a --n 10 --ton 1000 --gains 1.0,2.0 --ps-gain 2.0 > $L/p2a.log 2>&1
grep -v -i "warn\|falling" $L/p2a.log | sed -n '/reads (/,$p' > $W/tables_p2a.txt
echo "[chain_p2] done $(date)"
