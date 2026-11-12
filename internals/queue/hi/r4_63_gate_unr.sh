# round-4 route-keyed cell on unrouted rows by benchmark
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_r4/arrays_proc_g002c26_r4t2.npz --rows unr_bench_items.txt --out ev_r4_gate_unr --cells "ao_x1.25=0-63:64:1.25" "tE_x1.5=0-63:65:1.5:gate=krtE,nobias" "tE_x1.8=0-63:65:1.8:gate=krtE,nobias" "tE_x2.1=0-63:65:2.1:gate=krtE,nobias" > $L/r4_gate_unr.log 2>&1
$R ana_leak_bench.py ev_r4_gate_unr > $W/tables_r4_gate_unr.txt 2>&1
