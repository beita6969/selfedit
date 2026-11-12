# law-table row C_pix on the axes of the other rows, part 1: what the linear mixers alone and the full-attention mixers alone restore of the VP2-lattice effect (10 reads, every 6th of the list)
C=/workspace/fudan/mingdazhang/selfedit-int4; W=/data/fd/mingdazhang/selfedit/int/work; L=$W/logs; R=$C/run_x.sh; RC=$C/run_c.sh; T=/data/fd/mingdazhang/selfedit/int/traces; FT=$T/floor_signal_routed.json
awk 'NR%6==1' $W/c_pix/pix_items.txt > $W/c_pix/pix_test_items.txt
FI_MODE=vp2 $RC c_pix pix_run.py a1_patch.py --rows pix_test_items.txt --out a1 --sets mix > $W/c_pix/logs/a1.log 2>&1
$RC c_pix ana1.py a1 > $W/c_pix/tables_a1.txt 2>&1
