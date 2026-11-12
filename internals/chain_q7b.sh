#!/bin/bash
# fable_int Vesta, card B: P2 step 1 first (late image-attention boost, activation level), then the law-table queue of chain_q7.sh.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs
bash $C/chain_p2.sh > $L/chain_p2.log 2>&1
bash $C/chain_q7.sh > $L/chain_q7.log 2>&1
echo "[chain_q7b] done $(date)"
