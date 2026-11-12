# round-4 cell, route-keyed form (shifted route gates as chosen in round 3): arrays, dose and residual on the round-3 student's own held-out reads, leak by benchmark, specs with nulls
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
$R mk_krt.py cells_r4/arrays_proc_g002c26_r4.npz cells_r4/arrays_proc_g002c26_r4t2.npz D:8:0.15 E:8:0.25 > $L/r4_mk_krt.log 2>&1
FI_TRACES=$T/loop_r4.json FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_r4/arrays_proc_g002c26_r4t2.npz --rows proc_test_loop4_items.txt --out ev_r4_gate_own --cells ao_x1.25=0-63:64:1.25 tE_x1.5=0-63:65:1.5:gate=krtE,nobias tE_x1.8=0-63:65:1.8:gate=krtE,nobias tE_x2.1=0-63:65:2.1:gate=krtE,nobias tD_x1.5=0-63:65:1.5:gate=krtD,nobias > $L/r4_gate_own.log 2>&1
$R ana_round.py ev_r4_gate_own ev_r4_gate_own ao_x1.25=tE_x1.5 ao_x1.25=tE_x1.8 ao_x1.25=tE_x2.1 ao_x1.25=tD_x1.5 > $W/tables_r4_gate_own_pair.txt 2>&1
E=cells_r4/edits; V=$W/cells_r4
$R mkproc.py cells_r4/arrays_proc_g002c26_r4t2.npz neurons_free3.json $E $V/arrays_proc_g002c26_r4t2.npz "PR4a64tE_g002c26_+1.5=0-63:65:1.5:krtE:nobias" "PR4a64tE_g002c26_+1.8=0-63:65:1.8:krtE:nobias" "PR4a64tE_g002c26_+2.1=0-63:65:2.1:krtE:nobias" > $L/r4_mkproc_gate.log 2>&1
$R mkrand2.py $E/PR4a64tE_g002c26_+1.8.json cells_r4/arrays_proc_g002c26_r4t2.npz a3l4/fit cells_r4/arrays_proc_g002c26_ctl_r4t2.npz $V/arrays_proc_g002c26_ctl_r4t2.npz $E $E/PR4a64tE_g002c26_+1.5.json $E/PR4a64tE_g002c26_+2.1.json > $L/r4_mkrand2_gate.log 2>&1
for f in $W/$E/*PR4a64tE*.json; do ln -s $f $W/cells_proc/edits/$(basename $f); done
date > $L/r4_gate_specs_done
