#!/bin/bash
# fable_int Vesta, card A, RESUME of chain_p1a after the hindsight run: P1 = the counting procedure g002c26 written into weights. Rows, forcing capture (shard 0), neuron list, fit (rank 64, all blocks),
# functional restoration by rank / band / dose, then (after card B's captures) route key, specs, controls, leakage, weight-form check. New files only.
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; R=$C/run_x.sh; L=$W/logs
mkdir -p $L; export FI_BASE_NOPREFIX=1
TR=/data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json
# rows exist (first launch 01:46); this chain resumes the capture: finished reads are skipped
$R a3_inject.py --mode proc --rows proc_fit2_items.txt --ns 512 --out a3p/fit --shard 0/2 > $L/a3_a2.log 2>&1
$R nuse3.py proc_fit_items.txt proc_unrouted_items.txt $C/used_neurons.json neurons_free3.json > $L/nuse3_2.log 2>&1
until grep -q "^\[int\].*\(done\|FAILED\)" $L/a3_b2.log 2>/dev/null; do sleep 5; done
AR=cells_proc/arrays_proc_g002c26_v2.npz
$R fit3p.py --a3 a3p/fit --out cells_proc --xs "" --rmax 64 --device cuda --val-fold 1 --base-arrays $C/arrays_keys_v2.npz --arrays-out arrays_proc_g002c26_v2.npz > $L/fit3p.log 2>&1
$R b_fn.py --bf16 --arrays $AR --rows proc_test_items.txt --out ev_p1 --cells "r8_all=0-63:8:1" "r64_all=0-63:64:1" "r64_all_x1.25=0-63:64:1.25" "r64_full=0-63:64:1:blocks=full" "r64_full_x1.25=0-63:64:1.25:blocks=full" "r32_full=0-63:32:1:blocks=full" "r64_L35-63=35-63:64:1" "r64_L24-34=24-34:64:1" > $L/ev_p1.log 2>&1
$R ana1.py ev_p1 --b > $W/tables_p1.txt 2>&1
until [ -f $L/kcap_done ]; do sleep 5; done
ARR=cells_proc/arrays_proc_g002c26_v2r.npz; V=$W/cells_proc
$R rkey_fit.py kcap_r kcap_u $AR $ARR > $L/rkey.log 2>&1
$R mkproc.py $ARR neurons_free3.json cells_proc/edits $V/arrays_proc_g002c26_v2r.npz "PRa64_g002c26_+1=0-63:64:1:ks" "PRa64r_g002c26_+1=0-63:64:1:kr" "PRf64_g002c26_+1=0-63:64:1:ks:full" "PRf64r_g002c26_+1=0-63:64:1:kr:full" "PRf64_g002c26_+1.25=0-63:64:1.25:ks:full" "PRf64r_g002c26_+1.25=0-63:64:1.25:kr:full" > $L/mkproc.log 2>&1
E=cells_proc/edits
$R mkrand2.py $E/PRa64_g002c26_+1.json $ARR a3p/fit cells_proc/arrays_proc_g002c26_ctl_v2.npz $V/arrays_proc_g002c26_ctl_v2.npz $E $E/PRa64r_g002c26_+1.json $E/PRf64_g002c26_+1.json $E/PRf64r_g002c26_+1.json $E/PRf64_g002c26_+1.25.json $E/PRf64r_g002c26_+1.25.json > $L/mkrand2.log 2>&1
CT=cells_proc/arrays_proc_g002c26_ctl_v2.npz
$R b_fn.py --bf16 --arrays $ARR --ctl $CT --rows proc_test_items.txt --out ev_p1c --cells "r64_full=0-63:64:1:blocks=full" "r64_full_kr=0-63:64:1:blocks=full,gate=kr" "r64_all=0-63:64:1" "r64_all_kr=0-63:64:1:gate=kr" "r64_full_rand=0-63:64:1:blocks=full,var=rand" "r64_full_rot=0-63:64:1:blocks=full,var=rot" "r64_all_rand=0-63:64:1:var=rand" "r64_all_rot=0-63:64:1:var=rot" > $L/ev_p1c.log 2>&1
$R ana1.py ev_p1c --b > $W/tables_p1c.txt 2>&1
$R b_fn.py --bf16 --arrays $ARR --ctl $CT --rows proc_unrouted_items.txt --out ev_p1u --cells "r64_full=0-63:64:1:blocks=full" "r64_full_kr=0-63:64:1:blocks=full,gate=kr" "r64_all=0-63:64:1" "r64_all_kr=0-63:64:1:gate=kr" "r64_full_rand=0-63:64:1:blocks=full,var=rand" > $L/ev_p1u.log 2>&1
$R ana_leak.py ev_p1u ev_p1c > $W/tables_p1_leak.txt 2>&1
$R b_xb.py --mode proc --bf16 --rows proc_test_items.txt --limit 6 --out ev_w1 --specs "PRf64=$E/PRf64_g002c26_+1.json" "PRf64r=$E/PRf64r_g002c26_+1.json" "RAND_PRf64=$E/RAND__PRf64_g002c26_+1.json" "ROT_PRf64=$E/ROT__PRf64_g002c26_+1.json" > $L/ev_w1.log 2>&1
$R ana1.py ev_w1 --b > $W/tables_w1.txt 2>&1
$R b_xb.py --mode proc --bf16 --rows proc_unrouted_items.txt --limit 6 --out ev_w1u --specs "PRf64=$E/PRf64_g002c26_+1.json" "PRf64r=$E/PRf64r_g002c26_+1.json" > $L/ev_w1u.log 2>&1
$R ana_leak.py ev_w1u ev_w1 > $W/tables_w1_leak.txt 2>&1
$R b_xb.py --mode proc --bf16 --rows proc_test_items.txt --limit 3 --out ev_w2 --specs "PRa64=$E/PRa64_g002c26_+1.json" "PRa64r=$E/PRa64r_g002c26_+1.json" > $L/ev_w2.log 2>&1
$R ana1.py ev_w2 --b > $W/tables_w2.txt 2>&1
echo "[chain_p1r_a] done $(date)"
