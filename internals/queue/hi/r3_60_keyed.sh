# round-3 cell, route-keyed form (gate = the route key instead of the always-on stream key), steps 1 to 1.75: dose and residual on the round-2 student's own held-out reads, leak on
# unrouted reads. Purpose: a cell that leaves unrouted reads (and their length) alone at the same dose on routed rows.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
AR=cells_r3/arrays_proc_g002c26_r3.npz
CE='r64_all=0-63:64:1 r64_all_x1.25=0-63:64:1.25 r64_all_kr=0-63:64:1:gate=kr r64_all_kr_x1.25=0-63:64:1.25:gate=kr r64_all_kr_x1.5=0-63:64:1.5:gate=kr r64_all_kr_x1.75=0-63:64:1.75:gate=kr'
FI_TRACES=$T/loop_r3.json FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_loop3_items.txt --out ev_r3_keyed_own --cells $CE > $L/r3_keyed_own.log 2>&1
$R ana1.py ev_r3_keyed_own --b > $W/tables_r3_keyed_own.txt 2>&1
$R ana_round.py ev_r3_keyed_own ev_r3_keyed_own r64_all=r64_all_kr r64_all_x1.25=r64_all_kr_x1.5 r64_all_x1.25=r64_all_kr_x1.75 > $W/tables_r3_keyed_pair.txt 2>&1
$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_unrouted_items.txt --out ev_r3_keyed_unr --cells $CE > $L/r3_keyed_unr.log 2>&1
$R ana_leak.py ev_r3_keyed_unr ev_r3_keyed_own > $W/tables_r3_keyed_leak.txt 2>&1
