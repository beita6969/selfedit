# route key, second fit with BENEFIT labels: open only on the routed rows of the benchmark where the text helps on signal rows (vlmsbiased); the other routed rows (babyvision,
# charxiv, blink, lego) and all unrouted rows are negatives. Same captures as 73; merged into the round-4 arrays under a second file.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
mkdir -p $W/kcap_r2b $W/kcap_u2b
for d in kcap_r:t kcap_rl:l; do s=${d%%:*}; t=${d##*:}; for f in $W/$s/*.npz; do b=$(basename $f .npz); case $b in vlmsbiased__*) ln -s $f $W/kcap_r2b/${b}__$t.npz;; *) ln -s $f $W/kcap_u2b/${b}__r$t.npz;; esac; done; done
for f in $W/kcap_u/*.npz; do ln -s $f $W/kcap_u2b/$(basename $f); done; for f in $W/kcap_um/*.npz; do ln -s $f $W/kcap_u2b/$(basename $f .npz)__m.npz; done
$R rkey_fit.py kcap_r2b kcap_u2b cells_r4/arrays_proc_g002c26_r4.npz cells_r4/arrays_proc_g002c26_r4k2b.npz > $L/key2b_fit.log 2>&1
