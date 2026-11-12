# law-table row C_pix: where the forcing lives; leave-one-benchmark-out explained energy with the control
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json
$RC c_pix h_stats.py a3c/fit > $W/c_pix/tables_pix_stats_c.txt 2>&1
$RC c_pix fit_lobo2.py --a3 a3c/fit --out cells_c --tag pix --base-arrays $C/arrays_keys_v2.npz --rmax 64 --no-save > $W/c_pix/logs/fit_c.log 2>&1
$RC c_pix ana_pix.py a3c/fit cells_c/ev_pix.npz > $W/c_pix/summary_c.txt 2>&1
