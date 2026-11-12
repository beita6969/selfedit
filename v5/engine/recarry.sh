#!/bin/bash
# after the cancelled carry batches have been popped (done files written), resubmit them as carry2_* at priority 6 (below generation 5)
V4=/data/fd/mingdazhang/selfedit/v4
until gpu run "cd $V4/q/out && test -f carry_g004c12.done && test -f carry_g003c00.done && test -f carry_g003c16.done && test -f carry_g004c11.done" >/dev/null 2>&1; do sleep 30; done
echo "$(date +%H:%M:%S) cancelled batches closed"
gpu run "cd $V4/q/out && cat carry_g004c12.done carry_g003c00.done carry_g003c16.done carry_g004c11.done; cd $V4/q/in && for c in g004c12 g003c00 g003c16 g004c11; do sed 's/\"priority\": -\{0,1\}[0-9]*/\"priority\": 6/' carry_\$c.jsonl > .carry2_\$c.tmp && mv .carry2_\$c.tmp carry2_\$c.jsonl; echo carry2_\$c \$(wc -l < carry2_\$c.jsonl) \$(grep -o '\"priority\": [-0-9]*' carry2_\$c.jsonl | sort | uniq -c); done"
echo "$(date +%H:%M:%S) resubmitted"
