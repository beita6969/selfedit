# round-2 fit at rank 128 (the reserve knob, measured before it is needed): does a larger rank lower the residual on the unedited model's held-out reads?
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
$R fit3p.py --a3 a3pu/fit --out cells_proc2u128 --xs "" --rmax 128 --device cuda --val-fold 1 --base-arrays cells_proc/arrays_proc_g002c26_v2r.npz --arrays-out arrays_proc_g002c26_v3u128.npz > $L/fit3p_e1b_u128.log 2>&1
FI_TRACES=$FT FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_proc2u128/arrays_proc_g002c26_v3u128.npz --rows proc_test_floor_items.txt --out ev_r2_rank --cells "r64_all=0-63:64:1" "r96_all=0-63:96:1" "r128_all=0-63:128:1" "r128_full=0-63:128:1:blocks=full" > $L/r2_rank.log 2>&1
$R ana1.py ev_r2_rank --b > $W/tables_r2_rank.txt 2>&1
$R ana_round.py ev_r2_rank ev_r2_rank r64_all=r96_all r64_all=r128_all > $W/tables_r2_rank_pair.txt 2>&1
