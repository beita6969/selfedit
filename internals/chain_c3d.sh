#!/bin/bash
# fable_int Vesta, card C: E1b = the counting cells refitted against exposure (coordinator's order 04:20), then the law-table row C_pix.
# E1b: the forcing of the routed text is captured ALSO on reads generated WITHOUT the text (floor-only reads of the routed fit rows, 2 per row, first 1,000 thinking tokens); two fits:
# U = union with the text-generated reads, N = text-free reads only; evaluation on the held-out routed rows (text-free reads first), specs and nulls for both. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh
FT=/data/fd/mingdazhang/selfedit/int/traces/floor_signal_routed.json; export FI_BASE_NOPREFIX=1
sleep 20
FI_TRACES=$FT $R mk_items4.py $FT proc_fit_items.txt proc_fit_floor_items2.txt 2 150 > $L/e1b_rows.log 2>&1
FI_TRACES=$FT FI_MAX_REASON=1000 $R a3_inject.py --mode proc --rows proc_fit_floor_items2.txt --ns 256 --out a3pf/fit > $L/a3pf.log 2>&1
mkdir -p $W/a3pu/fit/samp
for f in $W/a3p/fit/*.npz; do b=$(basename $f .npz); ln -s $f $W/a3pu/fit/${b}__t.npz; ln -s $W/a3p/fit/samp/$b.npz $W/a3pu/fit/samp/${b}__t.npz; done
for f in $W/a3pf/fit/*.npz; do b=$(basename $f .npz); ln -s $f $W/a3pu/fit/${b}__f.npz; ln -s $W/a3pf/fit/samp/$b.npz $W/a3pu/fit/samp/${b}__f.npz; done
BASE=cells_proc/arrays_proc_g002c26_v2r.npz
$R fit3p.py --a3 a3pu/fit --out cells_proc2u --xs "" --rmax 64 --device cuda --val-fold 1 --base-arrays $BASE --arrays-out arrays_proc_g002c26_v3u.npz > $L/fit3p_e1b_u.log 2>&1
$R fit3p.py --a3 a3pf/fit --out cells_proc2n --xs "" --rmax 64 --device cuda --val-fold 1 --base-arrays $BASE --arrays-out arrays_proc_g002c26_v3n.npz > $L/fit3p_e1b_n.log 2>&1
CE='r64_full=0-63:64:1:blocks=full r64_all=0-63:64:1 r64_full_kr=0-63:64:1:blocks=full,gate=kr r64_full_rrot=0-63:64:1:blocks=full,var=rrot r64_full_wrot=0-63:64:1:blocks=full,var=wrot r64_all_rrot=0-63:64:1:var=rrot'
for v in u n; do
  FI_TRACES=$FT FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_proc2$v/arrays_proc_g002c26_v3$v.npz --rows proc_test_floor_items.txt --out ev_e1b_x$v --cells $CE > $L/ev_e1b_x$v.log 2>&1
  $R ana1.py ev_e1b_x$v --b > $W/tables_e1b_x$v.txt 2>&1
done
date > $L/e1b_textfree_done
for v in u n; do
  V=$(echo $v | tr un UN); AR=cells_proc2$v/arrays_proc_g002c26_v3$v.npz; E=cells_proc2$v/edits; VP=$W/cells_proc2$v
  A3=a3pu/fit; [ $v = n ] && A3=a3pf/fit
  $R mkproc.py $AR neurons_free3.json $E $VP/arrays_proc_g002c26_v3$v.npz "PR${V}f64_g002c26_+1=0-63:64:1:ks:full" "PR${V}a64_g002c26_+1=0-63:64:1:ks" "PR${V}f64r_g002c26_+1=0-63:64:1:kr:full" > $L/mkproc_e1b_$v.log 2>&1
  $R mkrand2.py $E/PR${V}a64_g002c26_+1.json $AR $A3 cells_proc2$v/arrays_proc_g002c26_ctl_v3$v.npz $VP/arrays_proc_g002c26_ctl_v3$v.npz $E $E/PR${V}f64_g002c26_+1.json > $L/mkrand2_e1b_$v.log 2>&1
done
date > $L/e1b_specs_done
for v in u n; do
  AR=cells_proc2$v/arrays_proc_g002c26_v3$v.npz
  $R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_test_items.txt --out ev_e1b_t$v --cells $CE > $L/ev_e1b_t$v.log 2>&1
  $R ana1.py ev_e1b_t$v --b > $W/tables_e1b_t$v.txt 2>&1
  $R b_fn5.py --mode proc --bf16 --arrays $AR --rows proc_unrouted_items.txt --out ev_e1b_u$v --cells "r64_full=0-63:64:1:blocks=full" "r64_all=0-63:64:1" "r64_full_kr=0-63:64:1:blocks=full,gate=kr" > $L/ev_e1b_u$v.log 2>&1
  $R ana_leak.py ev_e1b_u$v ev_e1b_x$v > $W/tables_e1b_leak_$v.txt 2>&1
done
date > $L/e1b_done
unset FI_BASE_NOPREFIX
$RC c_pix pix_run.py a3_inject2.py --mode vp2 --no-att --rows pix_items.txt --ns 192 --out a3b/fit > $W/c_pix/logs/a3b.log 2>&1
$RC c_pix h_stats.py a3b/fit > $W/c_pix/tables_pix_stats.txt 2>&1
$RC c_pix fit_lobo2.py --a3 a3b/fit --out cells --tag pix --base-arrays $C/arrays_keys_v2.npz --rmax 64 --no-save > $W/c_pix/logs/fit.log 2>&1
$RC c_pix ana_pix.py a3b/fit cells/ev_pix.npz > $W/c_pix/summary.txt 2>&1
echo "[chain_c3d] done $(date)"
