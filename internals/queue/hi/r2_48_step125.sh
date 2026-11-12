# round-2 all-block cell with step 1.25 (the write scaled; same units and neurons): spec and nulls, linked for the engine
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
AR=cells_proc2u/arrays_proc_g002c26_v3u.npz; E=cells_proc2u/edits; V=$W/cells_proc2u
$R mkproc.py $AR neurons_free3.json $E $V/arrays_proc_g002c26_v3u.npz "PRUa64_g002c26_+1.25=0-63:64:1.25:ks" > $L/mkproc_e1b_u125.log 2>&1
$R mkrand2.py $E/PRUa64_g002c26_+1.25.json $AR a3pf/fit cells_proc2u/arrays_proc_g002c26_ctl_v3u125.npz $V/arrays_proc_g002c26_ctl_v3u125.npz $E > $L/mkrand2_e1b_u125.log 2>&1
for f in $W/$E/*PRUa64_g002c26_+1.25.json; do ln -s $f $W/cells_proc/edits/$(basename $f); done
date > $L/r2_step125_done
