#!/bin/bash
# wait_cell.sh NAME BATCH,BATCH: wait for all .done, then cell_report vs W-null (full) and export TSVs to ~/selfedit/v4_reports/w_signal/ + batch_cells.json
V4=/data/fd/mingdazhang/selfedit/v4; ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4; N=$1; B=$2; REF=${3:-wnull_sig_s0,wnull_sig_s1}
while :; do n=0; t=0; for b in ${B//,/ }; do t=$((t+1)); gpu run "test -e $V4/q/out/$b.done" >/dev/null 2>&1 && n=$((n+1)); done; [ $n = $t ] && break; sleep 60; done
gpu run "cd $V4 && python3 $ED/cell_report.py $N $B --ref $REF; python3 $ED/w_export.py reports/w_signal ${B//,/ } >/dev/null"
cd ~/selfedit/v4_reports/w_signal; for b in ${B//,/ }; do gpu get $V4/reports/w_signal/$b.tsv . >/dev/null; done; python3 ~/selfedit/code-v4/engine_v4/batch_cells.py >/dev/null; echo exported ${B}
