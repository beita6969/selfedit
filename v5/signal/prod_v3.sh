#!/bin/bash
# v3 f32 production on Vesta (odd rows = --row-shard 1/2).  usage: prod_v3.sh MODE single CARD SHARD_I   |   prod_v3.sh MODE pair CARDA,CARDB
# MODE = fla (kernels) or torch (fallback via nofla.py)
MODE=$1; KIND=$2
S=/workspace/fudan/mingdazhang/selfedit-signal; O=/data/fd/mingdazhang/selfedit/v4/signal; TR=/data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json
MD=/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B; OUT=$O/scores_t005r_v3
PRE="unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=/data/fd/mingdazhang/selfedit/.hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/data/fd/mingdazhang/selfedit/.triton-sig; cd $O"
PY="$S/venv/bin/python -u"; [ "$MODE" = torch ] && PY="$S/venv/bin/python -u $S/nofla.py"
COMMON="--traces $TR --rows mixed --row-shard 1/2 --f32 --out $OUT --model $MD"
if [ "$KIND" = single ]; then
  c=$3; i=$4
  gpu bg --name selfedit-v4s-f32c$c --gpus $c "$PRE; $PY $S/score_worker_v3.py $COMMON --shard $i/${5:-4} --max-total-tokens 11000" | tail -1
else
  cs=$3; tag=${cs/,/}
  gpu bg --name selfedit-v4s-f32p$tag --gpus $cs "$PRE; $PY $S/score_worker_v3.py $COMMON --shard 0/1 --device cuda:0 --device2 cuda:1 --split-at 32 --retry-oom --reverse" | tail -1
fi
