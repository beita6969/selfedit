# round-1 full-attention cell (PRf64) on the reads IT generated (engine's E1 table reads, routed rows, samples 0-1): residual to the teacher on its own distribution, held-out routed rows;
# and the round-2 full-attention cell on the same reads (paired)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
EX=/data/fd/mingdazhang/selfedit/v4/scratch/v5_exports/e1loop_PRf64_g002c26_p1_loop_s01.jsonl.gz; TR=$T/loop_r1f.json
$R conv_floor_v.py $EX /data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json $TR > $L/r1f_prep.log 2>&1
FI_TRACES=$TR $R mk_items4.py $TR proc_test_items.txt proc_test_loop1f_items.txt 2 150 >> $L/r1f_prep.log 2>&1
FI_TRACES=$TR FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_proc/arrays_proc_g002c26_v2r.npz --rows proc_test_loop1f_items.txt --out ev_r1f_own_r1 --cells "r64_full=0-63:64:1:blocks=full" "r64_all=0-63:64:1" "r64_full_rrot=0-63:64:1:blocks=full,var=rrot" > $L/r1f_own_r1.log 2>&1
$R ana1.py ev_r1f_own_r1 --b > $W/tables_r1f_own_r1.txt 2>&1
FI_TRACES=$TR FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_proc2u/arrays_proc_g002c26_v3u.npz --rows proc_test_loop1f_items.txt --out ev_r1f_own_r2 --cells "r64_full=0-63:64:1:blocks=full" "r64_all=0-63:64:1" > $L/r1f_own_r2.log 2>&1
$R ana_round.py ev_r1f_own_r1 ev_r1f_own_r2 > $W/tables_r1f_own_pair.txt 2>&1
