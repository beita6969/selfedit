#!/bin/bash
# fable_int Vesta, iterated residual compile of the counting procedure g002c26: writes the high-priority queue tasks of one round.
# mk_round.sh R EXPORT PREV_ARRAYS "AGG"      R = round number (>= 3); EXPORT = the engine's loop reads of the previous round's cell (75 routed rows x 2, full reasoning, no text);
# PREV_ARRAYS = arrays of the previous round's cell (path relative to work/); AGG = the capture folders already in the aggregate with their tags, e.g. "a3p/fit:t a3pf/fit:f".
# Round R: (prep) convert the reads, row lists; (cap) forcing of the text on the previous student's reads, 3 shards; (fit) one rank-64 read per block refitted in closed form on the
# aggregate AGG + this round's reads; (ev) previous cell and new cell on the previous student's held-out reads, new cell on text-generated, unedited-model and unrouted reads;
# (specs) all-block and full-attention cells with RAND__ / ROT__ nulls, linked into cells_proc/edits/ for the engine. Tasks appear atomically (.need first, .sh by rename). New files only.
r=$1; EXPORT=$2; PREV=$3; AGG=$4
C=/workspace/fudan/mingdazhang/selfedit-int4; QS=$C/queue/hi; mkdir -p $QS
H='C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json; export FI_BASE_NOPREFIX=1'
TR="\$T/loop_r$r.json"; AR="cells_r$r/arrays_proc_g002c26_r$r.npz"
CE='"r64_all=0-63:64:1" "r64_full=0-63:64:1:blocks=full" "r64_all_rrot=0-63:64:1:var=rrot" "r64_all_wrot=0-63:64:1:var=wrot"'
put() { n=$1; need=$2; body=$3; [ -n "$need" ] && echo "$need" > $QS/$n.need; printf '%s\n%s\n' "$H" "$body" > $QS/.$n.tmp; mv $QS/.$n.tmp $QS/$n.sh; }
put r${r}_00_prep "" "\$R conv_floor_v.py $EXPORT /data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json $TR > \$L/r${r}_prep.log 2>&1
FI_TRACES=$TR \$R mk_items4.py $TR proc_fit_items.txt proc_fit_loop${r}_items.txt 2 150 >> \$L/r${r}_prep.log 2>&1
FI_TRACES=$TR \$R mk_items4.py $TR proc_test_items.txt proc_test_loop${r}_items.txt 2 150 >> \$L/r${r}_prep.log 2>&1"
for s in 0 1 2; do
put r${r}_10_cap$s "r${r}_00_prep" "FI_TRACES=$TR FI_MAX_REASON=1000 \$R a3_inject.py --mode proc --rows proc_fit_loop${r}_items.txt --ns 256 --shard $s/3 --out a3l$r/fit > \$L/r${r}_cap$s.log 2>&1"
done
LN="mkdir -p \$W/a3pu$r/fit/samp"
for at in $AGG a3l$r/fit:l$r; do d=${at%%:*}; t=${at##*:}
LN="$LN
for f in \$W/$d/*.npz; do b=\$(basename \$f .npz); ln -s \$f \$W/a3pu$r/fit/\${b}__$t.npz; ln -s \$W/$d/samp/\$b.npz \$W/a3pu$r/fit/samp/\${b}__$t.npz; done"
done
put r${r}_20_fit "r${r}_10_cap0 r${r}_10_cap1 r${r}_10_cap2" "$LN
\$R fit3p.py --a3 a3pu$r/fit --out cells_r$r --xs \"\" --rmax 64 --device cuda --val-fold 1 --base-arrays cells_proc/arrays_proc_g002c26_v2r.npz --arrays-out arrays_proc_g002c26_r$r.npz > \$L/r${r}_fit.log 2>&1"
put r${r}_30_evprev "r${r}_00_prep" "FI_TRACES=$TR FI_MAX_REASON=3000 \$R b_fn5.py --mode proc --bf16 --arrays $PREV --rows proc_test_loop${r}_items.txt --out ev_r${r}_prev_own --cells $CE > \$L/r${r}_evprev.log 2>&1
\$R ana1.py ev_r${r}_prev_own --b > \$W/tables_r${r}_prev_own.txt 2>&1"
put r${r}_40_evnew_own "r${r}_20_fit" "FI_TRACES=$TR FI_MAX_REASON=3000 \$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_loop${r}_items.txt --out ev_r${r}_new_own --cells $CE > \$L/r${r}_evnew_own.log 2>&1
\$R ana1.py ev_r${r}_new_own --b > \$W/tables_r${r}_new_own.txt 2>&1"
put r${r}_41_evnew_0 "r${r}_20_fit" "FI_TRACES=\$FT FI_MAX_REASON=3000 \$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_floor_items.txt --out ev_r${r}_new_0 --cells $CE > \$L/r${r}_evnew_0.log 2>&1
\$R ana1.py ev_r${r}_new_0 --b > \$W/tables_r${r}_new_0.txt 2>&1"
put r${r}_42_evnew_T "r${r}_20_fit" "\$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_items.txt --out ev_r${r}_new_T --cells $CE > \$L/r${r}_evnew_T.log 2>&1
\$R ana1.py ev_r${r}_new_T --b > \$W/tables_r${r}_new_T.txt 2>&1
\$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_unrouted_items.txt --out ev_r${r}_new_unr --cells \"r64_all=0-63:64:1\" \"r64_full=0-63:64:1:blocks=full\" \"r64_all_rrot=0-63:64:1:var=rrot\" > \$L/r${r}_evnew_unr.log 2>&1
\$R ana_leak.py ev_r${r}_new_unr ev_r${r}_new_T > \$W/tables_r${r}_leak.txt 2>&1"
put r${r}_50_specs "r${r}_20_fit" "E=cells_r$r/edits; V=\$W/cells_r$r
\$R mkproc.py $AR neurons_free3.json \$E \$V/arrays_proc_g002c26_r$r.npz \"PR${r}a64_g002c26_+1=0-63:64:1:ks\" \"PR${r}f64_g002c26_+1=0-63:64:1:ks:full\" > \$L/r${r}_mkproc.log 2>&1
\$R mkrand2.py \$E/PR${r}a64_g002c26_+1.json $AR a3l$r/fit cells_r$r/arrays_proc_g002c26_ctl_r$r.npz \$V/arrays_proc_g002c26_ctl_r$r.npz \$E \$E/PR${r}f64_g002c26_+1.json > \$L/r${r}_mkrand2.log 2>&1
for f in \$W/\$E/*PR${r}*.json; do ln -s \$f \$W/cells_proc/edits/\$(basename \$f); done
date > \$L/r${r}_specs_done"
ls $QS | grep "^r${r}_" | tr '\n' ' '; echo
