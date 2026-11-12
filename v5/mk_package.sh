#!/bin/bash
# assemble ~/selfedit/b5_restart/stage from ~/selfedit/code-v5 (copies only; nothing is deleted outside stage/)
set -e
SRC=/Users/admin/selfedit/code-v5; DST=/Users/admin/selfedit/b5_restart/stage
mkdir -p $DST/loop $DST/v5 $DST/harness $DST/engine_v4 $DST/data
cp -p $SRC/loop/*.py $SRC/loop/knobs.json $SRC/loop/run.sh $SRC/loop/*.txt $DST/loop/
cp -p $SRC/v5/*.py $DST/v5/
cp -p $SRC/harness/*.py $DST/harness/
cp -p $SRC/engine_v4/grade_mm.py $DST/engine_v4/
cp -p $SRC/data/learn.jsonl $DST/data/
cd $DST && find . -type f ! -name FILES.sha256 | sort | xargs shasum -a 256 > FILES.sha256
echo "files $(wc -l < FILES.sha256)"; grep -E "gen5.py|judge_v5.py|policy5.py|stats5.py|guess5.py|fake_dispatch.py|learn.jsonl|t_gen5.py|t_judge_v5.py|dry5.py" FILES.sha256 | cut -c1-16,65-
