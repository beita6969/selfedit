# round-2 cell, step size: the same units with the write scaled by 1.25 and 1.5, on the unedited model's held-out reads, the text-generated held-out reads and the unrouted reads
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
AR=cells_proc2u/arrays_proc_g002c26_v3u.npz; CE='r64_all=0-63:64:1 r64_all_x1.25=0-63:64:1.25 r64_all_x1.5=0-63:64:1.5 r64_full_x1.25=0-63:64:1.25:blocks=full r64_full_x1.5=0-63:64:1.5:blocks=full'
FI_TRACES=$FT FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_floor_items.txt --out ev_r2_dose_0 --cells $CE > $L/r2_dose_0.log 2>&1
$R ana1.py ev_r2_dose_0 --b > $W/tables_r2_dose_0.txt 2>&1
$R ana_round.py ev_r2_dose_0 ev_r2_dose_0 r64_all=r64_all_x1.25 r64_all=r64_all_x1.5 > $W/tables_r2_dose_0_pair.txt 2>&1
$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_items.txt --out ev_r2_dose_T --cells $CE > $L/r2_dose_T.log 2>&1
$R ana1.py ev_r2_dose_T --b > $W/tables_r2_dose_T.txt 2>&1
$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_unrouted_items.txt --out ev_r2_dose_unr --cells "r64_all=0-63:64:1" "r64_all_x1.25=0-63:64:1.25" "r64_all_x1.5=0-63:64:1.5" > $L/r2_dose_unr.log 2>&1
$R ana_leak.py ev_r2_dose_unr ev_r2_dose_0 > $W/tables_r2_dose_leak.txt 2>&1
