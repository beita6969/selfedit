# law-table row C_pix, forcing capture, shard 0 of 2 (stock lattice against VP2 lattice on the original image)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json
$RC c_pix pix_run.py a3_inject2.py --mode vp2 --no-att --rows pix_items.txt --ns 192 --shard 0/2 --out a3c/fit > $W/c_pix/logs/a3c_0.log 2>&1
