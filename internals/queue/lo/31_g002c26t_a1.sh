# law table: counting procedure on reads generated WITH the text (t005r): attention cuts, mixers, depth profile (held-out routed rows)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json
export FI_BASE_NOPREFIX=1 FI_MAX_REASON=1500; mkdir -p $W/c_g002c26t/logs
FI_MODE=proc $RC c_g002c26t a1_patch.py --rows $W/proc_test_items.txt --out a1 --sets cut,mix,res > $W/c_g002c26t/logs/a1.log 2>&1
$RC c_g002c26t ana1.py a1 > $W/c_g002c26t/tables_a1.txt 2>&1
