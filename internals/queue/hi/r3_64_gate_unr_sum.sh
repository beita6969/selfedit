# by-benchmark leak table of the gate variants
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
$R ana_leak_bench.py ev_r3_gate_unr > $W/tables_r3_gate_unr.txt 2>&1
