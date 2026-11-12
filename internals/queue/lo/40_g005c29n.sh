# law table: routed procedure g005c29 as a whole component on reads generated WITHOUT its text (floor-only reads of its rows)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json
export FI_BASE_NOPREFIX=1 FI_MAX_REASON=1500; mkdir -p $W/c_g005c29n/logs
EX=/data/fd/mingdazhang/selfedit/v4/scratch/v5_exports/e13_floor_signal_full_s0-7.jsonl.gz
$RC c_g005c29n conv_floor_parts.py $T/t007c_g005c29.json $EX $T/floor_g005c29.json > $W/c_g005c29n/logs/rows.log 2>&1
$RC c_g005c29n mk_items4.py $T/floor_g005c29.json $W/c_g005c29/proc_fit_items.txt proc_fit_items.txt 1 150 >> $W/c_g005c29n/logs/rows.log 2>&1
$RC c_g005c29n mk_items4.py $T/floor_g005c29.json $W/c_g005c29/proc_test_items.txt proc_test_items.txt 1 150 >> $W/c_g005c29n/logs/rows.log 2>&1
FI_TRACES=$T/floor_g005c29.json bash $C/chain_comp3.sh g005c29n proc $W/c_g005c29n/proc_fit_items.txt $W/c_g005c29n/proc_test_items.txt > $L/comp_g005c29n.log 2>&1
