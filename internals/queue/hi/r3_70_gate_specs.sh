# round-3 cell, route-keyed forms: specs with RAND__ / ROT__ nulls for the three candidate gates (65 gated units per block: 64 reads + the bias read through the constant key)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
E=cells_r3/edits; V=$W/cells_r3
$R mkproc.py cells_r3/arrays_proc_g002c26_r3t2.npz neurons_free3.json $E $V/arrays_proc_g002c26_r3t2.npz "PR3a64tD_g002c26_+1.5=0-63:65:1.5:krtD:nobias" "PR3a64tE_g002c26_+1.8=0-63:65:1.8:krtE:nobias" > $L/r3_mkproc_gate.log 2>&1
$R mkproc.py cells_r3/arrays_proc_g002c26_r3t.npz neurons_free3.json $E $V/arrays_proc_g002c26_r3t.npz "PR3a64tA_g002c26_+1=0-63:65:1:krtA:nobias" >> $L/r3_mkproc_gate.log 2>&1
$R mkrand2.py $E/PR3a64tD_g002c26_+1.5.json cells_r3/arrays_proc_g002c26_r3t2.npz a3l3/fit cells_r3/arrays_proc_g002c26_ctl_r3t2.npz $V/arrays_proc_g002c26_ctl_r3t2.npz $E $E/PR3a64tE_g002c26_+1.8.json > $L/r3_mkrand2_gate.log 2>&1
$R mkrand2.py $E/PR3a64tA_g002c26_+1.json cells_r3/arrays_proc_g002c26_r3t.npz a3l3/fit cells_r3/arrays_proc_g002c26_ctl_r3t.npz $V/arrays_proc_g002c26_ctl_r3t.npz $E >> $L/r3_mkrand2_gate.log 2>&1
for f in $W/$E/*PR3a64t*.json; do ln -s $f $W/cells_proc/edits/$(basename $f); done
date > $L/r3_gate_specs_done
