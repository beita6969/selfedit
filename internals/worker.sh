#!/bin/bash
# fable_int Vesta card worker: runs task scripts from the queue one at a time, high priority (queue/hi) before gap work (queue/lo), names in sort order.
# worker.sh NAME [WAIT_LOG WAIT_PATTERN]   (waits for the pattern in the log of the chain that still holds the card, then starts)
# A task NAME.sh may have NAME.need (names of tasks that must be done first) and NAME.only (workers allowed to take it). Claims, done markers and task logs: work/queue/.
# Stop after the current task: touch work/queue/stop_NAME (or stop_all). Nothing is deleted.
C=/workspace/fudan/mingdazhang/selfedit-int4; QS=$C/queue; Q=/data/fd/mingdazhang/selfedit/int/work/queue; ME=$1
mkdir -p $Q/claim $Q/done $Q/log
if [ -n "$2" ]; then until grep -q "$3" "$2" 2>/dev/null; do sleep 10; done; fi
echo "[worker $ME] $(date) up"
while [ ! -f $Q/stop_$ME ] && [ ! -f $Q/stop_all ]; do
  t=""
  for d in hi lo; do
    for f in $(ls $QS/$d/*.sh 2>/dev/null | sort); do
      n=$(basename $f .sh); [ -d $Q/claim/$n ] && continue
      if [ -f $QS/$d/$n.only ] && ! grep -qw "$ME" $QS/$d/$n.only; then continue; fi
      ok=1; if [ -f $QS/$d/$n.need ]; then for x in $(cat $QS/$d/$n.need); do [ -f $Q/done/$x ] || ok=0; done; fi
      [ $ok = 1 ] || continue
      if mkdir $Q/claim/$n 2>/dev/null; then t=$f; break 2; fi
    done
  done
  if [ -z "$t" ]; then sleep 5; continue; fi
  n=$(basename $t .sh); echo "[worker $ME] $(date) start $n"; echo $ME > $Q/claim/$n/by
  bash $t > $Q/log/$n.log 2>&1; rc=$?; echo "[worker $ME] $(date) end $n rc=$rc"; echo "$ME rc=$rc $(date)" > $Q/done/$n
done
echo "[worker $ME] $(date) stopped"
