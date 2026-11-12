#!/bin/bash
# wait_judge.sh BATCH...: wait until all batches have .done, export JUDGE TSVs to ~/selfedit/v4_reports/w_judge/ (never w_signal)
V4=/data/fd/mingdazhang/selfedit/v4; ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4
while :; do n=0; for b in "$@"; do gpu run "test -e $V4/q/out/$b.done" >/dev/null 2>&1 && n=$((n+1)); done; [ $n = $# ] && break; sleep 60; done
gpu run "cd $V4 && python3 $ED/w_export_j.py reports/w_judge $*"
mkdir -p ~/selfedit/v4_reports/w_judge; cd ~/selfedit/v4_reports/w_judge
for b in "$@"; do gpu get $V4/reports/w_judge/$b.tsv . | tail -1; done
for b in "$@"; do echo "$b rows $(($(wc -l < $b.tsv)-1)) errors $(awk -F'\t' 'NR>1 && $10!=""' $b.tsv | wc -l) cell $(awk -F'\t' 'NR==2{print $2}' $b.tsv)"; done
date
