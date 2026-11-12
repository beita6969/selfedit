#!/bin/bash
# fable_int Vesta: E1b = the counting cells refitted against exposure. The forcing of the routed text is captured ALSO on reads that were generated WITHOUT the text (floor-only reads of the
# routed fit rows), and the state read is fitted on the union with the text-generated reads; route key refitted with the text-free routed streams included. Evaluation on held-out routed
# rows, both kinds of read; specs and nulls. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh
FT=/data/fd/mingdazhang/selfedit/int/traces/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
[ -f $W/proc_test_floor_items.txt ] || FI_TRACES=$FT $R mk_items4.py $FT proc_test_items.txt proc_test_floor_items.txt 2 150 > $L/expo_rows.log 2>&1
FI_TRACES=$FT $R mk_items4.py $FT proc_fit_items.txt proc_fit_floor_items.txt 1 150 > $L/e1b_rows.log 2>&1
FI_TRACES=$FT FI_MAX_REASON=3000 $R a3_inject.py --mode proc --rows proc_fit_floor_items.txt --ns 384 --out a3pf/fit > $L/a3pf.log 2>&1
mkdir -p $W/a3pu/fit/samp
for f in $W/a3p/fit/*.npz; do b=$(basename $f .npz); ln -s $f $W/a3pu/fit/${b}__t.npz; ln -s $W/a3p/fit/samp/$b.npz $W/a3pu/fit/samp/${b}__t.npz; done
for f in $W/a3pf/fit/*.npz; do b=$(basename $f .npz); ln -s $f $W/a3pu/fit/${b}__f.npz; ln -s $W/a3pf/fit/samp/$b.npz $W/a3pu/fit/samp/${b}__f.npz; done
$R fit3p.py --a3 a3pu/fit --out cells_proc2 --xs "" --rmax 64 --device cuda --val-fold 1 --base-arrays cells_proc/arrays_proc_g002c26_v2r.npz --arrays-out arrays_proc_g002c26_v3.npz > $L/fit3p_e1b.log 2>&1
AR=cells_proc2/arrays_proc_g002c26_v3.npz
CE='r64_full=0-63:64:1:blocks=full r64_full_kr=0-63:64:1:blocks=full,gate=kr r64_all=0-63:64:1 r64_all_kr=0-63:64:1:gate=kr r64_full_rrot=0-63:64:1:blocks=full,var=rrot r64_full_wrot=0-63:64:1:blocks=full,var=wrot'
FI_TRACES=$FT FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_floor_items.txt --out ev_e1b_x --cells $CE > $L/ev_e1b_x.log 2>&1
$R ana1.py ev_e1b_x --b > $W/tables_e1b_x.txt 2>&1
$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_items.txt --out ev_e1b_t --cells $CE > $L/ev_e1b_t.log 2>&1
$R ana1.py ev_e1b_t --b > $W/tables_e1b_t.txt 2>&1
$R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_unrouted_items.txt --out ev_e1b_u --cells "r64_full=0-63:64:1:blocks=full" "r64_full_kr=0-63:64:1:blocks=full,gate=kr" "r64_all=0-63:64:1" > $L/ev_e1b_u.log 2>&1
$R ana_leak.py ev_e1b_u ev_e1b_x > $W/tables_e1b_leak.txt 2>&1
V=$W/cells_proc2; E=cells_proc2/edits
$R mkproc.py $AR neurons_free3.json $E $V/arrays_proc_g002c26_v3.npz "PRXf64_g002c26_+1=0-63:64:1:ks:full" "PRXa64_g002c26_+1=0-63:64:1:ks" "PRXf64r_g002c26_+1=0-63:64:1:kr:full" > $L/mkproc_e1b.log 2>&1
$R mkrand2.py $E/PRXf64_g002c26_+1.json $AR a3pu/fit cells_proc2/arrays_proc_g002c26_ctl_v3.npz $V/arrays_proc_g002c26_ctl_v3.npz $E $E/PRXa64_g002c26_+1.json $E/PRXf64r_g002c26_+1.json > $L/mkrand2_e1b.log 2>&1
echo "[chain_e1b] done $(date)"
