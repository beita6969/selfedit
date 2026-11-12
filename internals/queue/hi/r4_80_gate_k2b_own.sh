# round-4 cell with the refitted route key (k2b: k2 = route labels, k2b = benefit labels): gates, dose and residual on the round-3 student's own held-out routed reads, by benchmark
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
$R mk_krt.py cells_r4/arrays_proc_g002c26_r4k2b.npz cells_r4/arrays_proc_g002c26_r4k2bt.npz E:8:0.25 F:16:0.5 > $L/r4_mk_krt_k2b.log 2>&1
$R keyz_bench.py cells_r4/arrays_proc_g002c26_r4k2b.npz kcap_r2 kcap_u2 8:0.25 16:0.5 > $W/tables_r4_keyz_k2b.txt 2>&1
FI_TRACES=$T/loop_r4.json FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_r4/arrays_proc_g002c26_r4k2bt.npz --rows proc_test_loop4_items.txt --out ev_r4_k2b_own --cells ao_x1.25=0-63:64:1.25 kr_x1.5=0-63:64:1.5:gate=kr tE_x1.5=0-63:65:1.5:gate=krtE,nobias tE_x1.8=0-63:65:1.8:gate=krtE,nobias tF_x1.25=0-63:65:1.25:gate=krtF,nobias tF_x1.5=0-63:65:1.5:gate=krtF,nobias > $L/r4_k2b_own.log 2>&1
$R ana_round.py ev_r4_k2b_own ev_r4_k2b_own ao_x1.25=kr_x1.5 ao_x1.25=tE_x1.5 ao_x1.25=tE_x1.8 ao_x1.25=tF_x1.25 ao_x1.25=tF_x1.5 > $W/tables_r4_k2b_own_pair.txt 2>&1
$R ana_gate_bench.py ev_r4_k2b_own > $W/tables_r4_k2b_own_bench.txt 2>&1
