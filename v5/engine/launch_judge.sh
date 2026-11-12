#!/bin/bash
# usage (on the Mac): launch_judge.sh DOSE_TAG [PRIO]   e.g. launch_judge.sh p075 -10
# copies the prepared XS JUDGE shards xsj_all_<DOSE_TAG>_s0..2 into q/in at PRIO (default -10) and records t_submit.
D=$1; P=${2:--10}; V4=/data/fd/mingdazhang/selfedit/v4
gpu run "cd $V4 && for i in 0 1 2; do b=xsj_all_${D}_s\$i; sed -E 's/\"priority\": -?[0-9]+/\"priority\": $P/' scratch/ext/judge_ready/\$b.jsonl > q/in/\$b.jsonl.tmp && mv q/in/\$b.jsonl.tmp q/in/\$b.jsonl && python3 -c \"import json,time; open('scratch/ext/submit_times.jsonl','a').write(json.dumps(dict(batch='\$b', t_submit=round(time.time(),1), n=sum(1 for _ in open('q/in/\$b.jsonl'))))+'\\\\n')\" && echo \"\$b launched at prio $P\"; done; tail -3 scratch/ext/submit_times.jsonl"
