# specs with RAND__ / ROT__ nulls for the candidate gates and steps of the round-4 cell with the refitted key (k2b)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
E=cells_r4/edits; VV=$W/cells_r4; AR=cells_r4/arrays_proc_g002c26_r4k2bt.npz
$R mkproc.py $AR neurons_free3.json $E $VV/arrays_proc_g002c26_r4k2bt.npz "PR4a64K2BE_g002c26_+1.5=0-63:65:1.5:krtE:nobias" "PR4a64K2BE_g002c26_+1.8=0-63:65:1.8:krtE:nobias" "PR4a64K2BF_g002c26_+1.25=0-63:65:1.25:krtF:nobias" "PR4a64K2BF_g002c26_+1.5=0-63:65:1.5:krtF:nobias" > $L/r4_mkproc_k2b.log 2>&1
$R mkrand2.py $E/PR4a64K2BE_g002c26_+1.8.json $AR a3l4/fit cells_r4/arrays_proc_g002c26_ctl_r4k2bt.npz $VV/arrays_proc_g002c26_ctl_r4k2bt.npz $E $E/PR4a64K2BE_g002c26_+1.5.json $E/PR4a64K2BF_g002c26_+1.25.json $E/PR4a64K2BF_g002c26_+1.5.json > $L/r4_mkrand2_k2b.log 2>&1
for f in $W/$E/*PR4a64K2B[EF]_*.json; do ln -s $f $W/cells_proc/edits/$(basename $f); done
date > $L/r4_specs_k2b_done
