# round 2 cell (fit u): restoration on the text-generated held-out reads
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
$R b_fn5.py --mode proc --bf16 --arrays cells_proc2u/arrays_proc_g002c26_v3u.npz --rows proc_test_items.txt --out ev_r2u_T --cells "r64_all=0-63:64:1" "r64_full=0-63:64:1:blocks=full" "r64_all_rrot=0-63:64:1:var=rrot" "r64_all_wrot=0-63:64:1:var=wrot" > $L/r2u_evT.log 2>&1
$R ana1.py ev_r2u_T --b > $W/tables_r2u_T.txt 2>&1
