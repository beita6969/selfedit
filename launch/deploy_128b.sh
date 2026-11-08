#!/bin/bash
# Deploy the frozen 128b code from res_driver_128 into live code-xf. Backup first, copy by temp file + rename, verify sha256. Nothing is deleted.
# usage: bash deploy_128b.sh          (dry: prints what it would do)
#        bash deploy_128b.sh --apply
set -euo pipefail
N=/workspace/fudan/mingdazhang/selfedit/res_driver_128
X=/workspace/fudan/mingdazhang/selfedit/code-xf
FILES="rsi/hboost.py rsi/mm_eval.py rsi/mm_gate.py rsi/res_fuse.py mm/grade_mm.py dry/res_replay.py dry/res_chain.py dry/res_grid.py dry/res_grid_panel.py dry/res_gw.py dry/res_job.py"
TS=$(date +%Y%m%d_%H%M%S)
BK=$X/_backup_128b_$TS
APPLY=${1:-}
echo "deploy 128b  $TS  apply=${APPLY:-no}"
for f in $FILES; do
  a=$(sha256sum $N/$f | cut -d' ' -f1)
  b=$([ -f $X/$f ] && sha256sum $X/$f | cut -d' ' -f1 || echo absent)
  echo "$f new=${a:0:16} live=${b:0:16} $([ "$a" = "$b" ] && echo same || echo change)"
done
[ "$APPLY" = "--apply" ] || { echo "dry only; rerun with --apply"; exit 0; }
mkdir -p $BK
for f in $FILES; do
  if [ -f $X/$f ]; then mkdir -p $BK/$(dirname $f); cp -p $X/$f $BK/$f; fi
done
( cd $BK && find . -type f -exec sha256sum {} \; ) > $BK/MANIFEST.sha256
echo "backup: $BK ($(wc -l < $BK/MANIFEST.sha256) files)"
for f in $FILES; do
  mkdir -p $X/$(dirname $f)
  cp -p $N/$f $X/$f.new128b
  mv -f $X/$f.new128b $X/$f
done
bad=0
for f in $FILES; do
  a=$(sha256sum $N/$f | cut -d' ' -f1); b=$(sha256sum $X/$f | cut -d' ' -f1)
  [ "$a" = "$b" ] && echo "ok $f ${b:0:16}" || { echo "MISMATCH $f"; bad=1; }
done
( cd $X && sha256sum $FILES ) > $BK/DEPLOYED.sha256
echo "deployed; restore = copy files from $BK back (cp -p), nothing was removed"
exit $bad
