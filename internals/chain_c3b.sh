#!/bin/bash
# fable_int Vesta, card C after the E1 chain: (x) E1 exposure check as soon as the floor-only reads are converted: the counting cells on reads of the held-out routed rows that were generated
# WITHOUT the text; then the rest of P2 step 1 (reads 7-10); then the law-table row C_pix (stock lattice against the VP2 lattice on the original image). New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh
FT=/data/fd/mingdazhang/selfedit/int/traces/floor_signal_routed.json
expo() {
  [ -f $FT ] || return 1
  [ -f $L/expo_started ] && return 0
  date > $L/expo_started
  FI_TRACES=$FT $R mk_items4.py $FT proc_test_items.txt proc_test_floor_items.txt 2 150 > $L/expo_rows.log 2>&1
  FI_TRACES=$FT FI_BASE_NOPREFIX=1 FI_MAX_REASON=3000 $R b_fn5.py --mode proc --bf16 --arrays cells_proc/arrays_proc_g002c26_v2r.npz --ctl cells_proc/arrays_proc_g002c26_ctl_v2.npz --rows proc_test_floor_items.txt --out ev_p1x --cells "r64_full=0-63:64:1:blocks=full" "r64_full_kr=0-63:64:1:blocks=full,gate=kr" "r64_all=0-63:64:1" "r64_all_kr=0-63:64:1:gate=kr" "r64_full_rand=0-63:64:1:blocks=full,var=rand" "r64_all_rand=0-63:64:1:var=rand" > $L/ev_p1x.log 2>&1
  $R ana1.py ev_p1x --b > $W/tables_p1x.txt 2>&1
  $R ana_leak.py ev_p1x ev_p1c > $W/tables_p1x_act.txt 2>&1
  date > $L/expo_done
}
until grep -q "chain_p1u\] done" /data/fd/mingdazhang/gpu-logs/selfedit-int-p1u.log 2>/dev/null; do sleep 10; done
expo
$R p2_reopen2.py --out p2a --n 10 --ton 1000 --gains 1.0,2.0 --ps-gain 2.0 > $L/p2a_2.log 2>&1
grep -v -i "warn\|falling" $L/p2a_2.log | sed -n '/reads (/,$p' > $W/tables_p2a_10.txt
$R ana_p2.py p2a 1000 > $W/tables_p2a_new.txt 2>&1
expo
mkdir -p $W/c_pix/logs
$RC c_pix mk_rows_pix.py /data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json pix_items.txt 10 > $W/c_pix/logs/rows.log 2>&1
$RC c_pix pix_run.py a3_inject2.py --mode vp2 --no-att --rows pix_items.txt --ns 192 --out a3/fit > $W/c_pix/logs/a3.log 2>&1
$RC c_pix h_stats.py a3/fit > $W/c_pix/tables_pix_stats.txt 2>&1
$RC c_pix fit_lobo2.py --a3 a3/fit --out cells --tag pix --base-arrays $C/arrays_keys_v2.npz --rmax 64 --no-save > $W/c_pix/logs/fit.log 2>&1
$RC c_pix ana_pix.py a3/fit cells/ev_pix.npz > $W/c_pix/summary.txt 2>&1
expo
echo "[chain_c3b] done $(date)"
