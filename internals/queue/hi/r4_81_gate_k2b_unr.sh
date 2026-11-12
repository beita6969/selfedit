# the same gates (k2b) on unrouted rows by benchmark, and specs with nulls for every candidate gate and step
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_r4/arrays_proc_g002c26_r4k2bt.npz --rows unr_bench_items.txt --out ev_r4_k2b_unr --cells kr_x1.5=0-63:64:1.5:gate=kr tE_x1.5=0-63:65:1.5:gate=krtE,nobias tE_x1.8=0-63:65:1.8:gate=krtE,nobias tF_x1.25=0-63:65:1.25:gate=krtF,nobias tF_x1.5=0-63:65:1.5:gate=krtF,nobias > $L/r4_k2b_unr.log 2>&1
$R ana_leak_bench.py ev_r4_k2b_unr > $W/tables_r4_k2b_unr.txt 2>&1
