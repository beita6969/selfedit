# gap round: the aggregate of round 2 plus the reads the ROUND-1 full-attention cell generated (fit rows; captured in the gap): does adding student-generated reads lower the
# residual on student-generated held-out reads? (the claim of the iterated method, tested on data in hand while the round-2 loop reads are awaited)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
mkdir -p $W/a3puv/fit/samp
for at in a3p/fit:t a3pf/fit:f a3l1f/fit:l1f; do d=${at%%:*}; t=${at##*:}; for f in $W/$d/*.npz; do b=$(basename $f .npz); ln -s $f $W/a3puv/fit/${b}__$t.npz; ln -s $W/$d/samp/$b.npz $W/a3puv/fit/samp/${b}__$t.npz; done; done
$R fit3p.py --a3 a3puv/fit --out cells_r2v --xs "" --rmax 64 --device cuda --val-fold 1 --base-arrays cells_proc/arrays_proc_g002c26_v2r.npz --arrays-out arrays_proc_g002c26_r2v.npz > $L/r2v_fit.log 2>&1
