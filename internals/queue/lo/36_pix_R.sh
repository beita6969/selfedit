# law-table row C_pix, part 2: restoration R by state-read units fitted on the other 50 reads (rank 64 all blocks), with controls, on the same 10 reads
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json
cd $W/c_pix; mkdir -p a3c50/fit/samp
awk 'NR%6==1 {gsub("/","__",$1); print $1 "__s" $2 ".npz"}' pix_items.txt > pix_test_files.txt
for f in a3c/fit/*.npz; do b=$(basename $f); grep -qx "$b" pix_test_files.txt || { ln -s $W/c_pix/a3c/fit/$b a3c50/fit/$b; ln -s $W/c_pix/a3c/fit/samp/$b a3c50/fit/samp/$b; }; done
$RC c_pix fit3p.py --a3 a3c50/fit --out cells_fit --xs "" --rmax 64 --device cuda --val-fold 1 --base-arrays $C/arrays_keys_v2.npz --arrays-out arrays_pix.npz > $W/c_pix/logs/fit3p.log 2>&1
$RC c_pix pix_run.py b_fn5.py --mode vp2 --bf16 --arrays cells_fit/arrays_pix.npz --rows pix_test_items.txt --out ev --cells "r8_all=0-63:8:1" "r64_all=0-63:64:1" "r64_full=0-63:64:1:blocks=full" "r64_lin=0-63:64:1:blocks=lin" "r64_all_wrot=0-63:64:1:var=wrot" "r64_all_rrot=0-63:64:1:var=rrot" > $W/c_pix/logs/ev.log 2>&1
$RC c_pix ana1.py ev --b > $W/c_pix/tables_R.txt 2>&1
