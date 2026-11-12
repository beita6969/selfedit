# route key, second fit: ridge on the union (routed: text-generated streams + the round-3 cell's own streams; unrouted: old + new rows); merged into the round-4 arrays
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
mkdir -p $W/kcap_r2 $W/kcap_u2
for f in $W/kcap_r/*.npz; do ln -s $f $W/kcap_r2/$(basename $f .npz)__t.npz; done; for f in $W/kcap_rl/*.npz; do ln -s $f $W/kcap_r2/$(basename $f .npz)__l.npz; done
for f in $W/kcap_u/*.npz; do ln -s $f $W/kcap_u2/$(basename $f); done; for f in $W/kcap_um/*.npz; do ln -s $f $W/kcap_u2/$(basename $f .npz)__m.npz; done
$R rkey_fit.py kcap_r2 kcap_u2 cells_r4/arrays_proc_g002c26_r4.npz cells_r4/arrays_proc_g002c26_r4k2.npz > $L/key2_fit.log 2>&1
