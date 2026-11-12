# law table: counting procedure on reads generated WITHOUT the text: attention cuts, mixers, depth profile (held-out routed rows, one read per row)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json
export FI_BASE_NOPREFIX=1 FI_MAX_REASON=1500; mkdir -p $W/c_g002c26x/logs
FI_TRACES=$FT FI_MODE=proc $RC c_g002c26x a1_patch.py --rows $W/proc_test_floor_items.txt --shard 0/2 --out a1 --sets cut,mix,res > $W/c_g002c26x/logs/a1.log 2>&1
FI_TRACES=$FT $RC c_g002c26x ana1.py a1 > $W/c_g002c26x/tables_a1.txt 2>&1
