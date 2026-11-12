# route key, second fit (the key learns the streams the artefact will see): row lists. Routed: the reads the round-3 cell generated (all routed rows); unrouted: rows not yet used for
# the key, except the 37 reads kept as the by-benchmark test set.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
cat $W/proc_fit_loop4_items.txt $W/proc_test_loop4_items.txt > $W/key2_routed_items.txt
$R mk_unr_bench.py /data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json key2_unr_items.txt 40 proc_unrfit_items.txt proc_unrouted_items.txt unr_bench_items.txt > $L/key2_prep.log 2>&1
