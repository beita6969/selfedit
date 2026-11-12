#!/bin/bash
# run.sh RUN_DIR [extra gen.py args]: one gen.py process per generation, so code uploaded between generations takes effect at the
# next boundary (coordinator 11-10: no pause between generations, Stage 2 switches on at the first boundary where it is ready).
# Stops at the generation boundary when RUN_DIR/STOP exists; three failures in a row stop it.
RUN=$1; shift
fails=0
while [ ! -e "$RUN/STOP" ]; do
  python -u /workspace/fudan/mingdazhang/selfedit-v4/loop/gen.py --run "$RUN" --gens 1 "$@"
  rc=$?
  if [ $rc -ne 0 ]; then
    fails=$((fails+1)); echo "$(date +%H:%M:%S) gen.py exit $rc (failure $fails in a row)"
    [ $fails -ge 3 ] && { echo "$(date +%H:%M:%S) three failures in a row: wrapper stops"; exit 1; }
    sleep 60
  else
    fails=0
  fi
done
echo "$(date +%H:%M:%S) STOP file: wrapper ends at the generation boundary"
