# round-3 cell, shifted linear route gates (slope 8, thresholds 0.15 and 0.25: the route key minus a constant, so that unrouted streams sit at zero while routed streams keep
# a near-proportional gate): dose and residual on the round-2 student's own held-out routed reads
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
FI_TRACES=$T/loop_r3.json FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_r3/arrays_proc_g002c26_r3t2.npz --rows proc_test_loop3_items.txt --out ev_r3_gate2_own --cells kr_x1.5=0-63:64:1.5:gate=kr tD_x1.5=0-63:65:1.5:gate=krtD,nobias tD_x1.8=0-63:65:1.8:gate=krtD,nobias tD_x2.1=0-63:65:2.1:gate=krtD,nobias tE_x1.8=0-63:65:1.8:gate=krtE,nobias tE_x2.2=0-63:65:2.2:gate=krtE,nobias tE_x2.6=0-63:65:2.6:gate=krtE,nobias > $L/r3_gate2_own.log 2>&1
$R ana1.py ev_r3_gate2_own --b > $W/tables_r3_gate2_own.txt 2>&1
$R ana_round.py ev_r3_gate2_own ev_r3_gate2_own kr_x1.5=tD_x1.5 kr_x1.5=tD_x1.8 kr_x1.5=tD_x2.1 kr_x1.5=tE_x1.8 kr_x1.5=tE_x2.2 kr_x1.5=tE_x2.6 > $W/tables_r3_gate2_own_pair.txt 2>&1
