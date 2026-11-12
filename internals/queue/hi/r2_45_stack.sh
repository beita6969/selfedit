# round 2 cell on the stacked base, teacher-forced: the same held-out text-free reads with every context tokenised by edit 1 (vp2m512: VP2 tower, 2,048-4,096 image tokens on the original image)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
export FI_VP_MODEL=/data/fd/mingdazhang/selfedit/v5/wckpt/vp2m512 FI_PIX_ALL=1
FI_TRACES=$FT FI_MAX_REASON=3000 $R pix_run2.py b_fn5.py --mode proc --bf16 --arrays cells_proc2u/arrays_proc_g002c26_v3u.npz --rows proc_test_floor_items.txt --out ev_stack_r2 --cells "r64_all=0-63:64:1" "r64_full=0-63:64:1:blocks=full" "r64_all_rrot=0-63:64:1:var=rrot" > $L/r2_stack.log 2>&1
$R ana1.py ev_stack_r2 --b > $W/tables_stack_r2.txt 2>&1
$R ana_round.py ev_e1b_xu ev_stack_r2 > $W/tables_stack_r2_pair.txt 2>&1
