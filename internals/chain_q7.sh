#!/bin/bash
# fable_int Vesta, card B queue after the hindsight chain: law-table components: the proposer's always-on sentence, then the two t007c entries (rows where the entry is the only routed procedure;
# first 1,500 thinking tokens of each read; g005c23 on rows whose prompt fits 6,500 tokens, 3 reads per row).
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; T=/data/fd/mingdazhang/selfedit/int/traces
bash $C/chain_comp3.sh ao1 pre@$C/ao1.txt > $L/comp_ao1.log 2>&1
export FI_BASE_NOPREFIX=1 FI_MAX_REASON=1500
FI_TRACES=$T/t007c_g005c29.json bash $C/chain_comp3.sh g005c29 proc $W/c_g005c29/proc_fit_items.txt $W/c_g005c29/proc_test_items.txt > $L/comp_g005c29.log 2>&1
(cd $W/c_g005c23b && FI_GEN_HI=40000 FI_PROMPT_HI=6500 FI_TRACES=$T/t007c_g005c23.json bash $C/run_c.sh c_g005c23b mk_rows_v2.py $T/t007c_g005c23.json proc 3 > logs/rows.log 2>&1; FI_TRACES=$T/t007c_g005c23.json bash $C/run_c.sh c_g005c23b mk_items3.py $T/t007c_g005c23.json proc_test_ids.txt proc_test3_items.txt 3 6500 >> logs/rows.log 2>&1)
FI_TRACES=$T/t007c_g005c23.json bash $C/chain_comp3.sh g005c23b proc $W/c_g005c23b/proc_fit2_items.txt $W/c_g005c23b/proc_test3_items.txt > $L/comp_g005c23b.log 2>&1
echo "[chain_q7] done $(date)"
