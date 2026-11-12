#!/bin/bash
# fable_int on Vesta (inside a gpu bg job; the card is the one given by --gpus): run_x.sh SCRIPT ARGS...
# code /workspace/fudan/mingdazhang/selfedit-int, outputs /data/fd/mingdazhang/selfedit/int/work (cwd), caches under /data/fd/mingdazhang/selfedit/int/cache. Nothing is deleted or overwritten.
SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; C=/workspace/fudan/mingdazhang/selfedit-int4; W=$D/int
. $SE/venv/bin/activate
unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset $(env | grep -o '^TRITON_[A-Z_]*')
mkdir -p $W/work/logs $W/cache/hf $W/cache/xdg $W/cache/tmp $W/cache/nv $W/cache/triton $W/cache/inductor
export MPLBACKEND=Agg HF_HOME=$W/cache/hf XDG_CACHE_HOME=$W/cache/xdg TMPDIR=$W/cache/tmp CUDA_CACHE_PATH=$W/cache/nv TRITON_CACHE_DIR=$W/cache/triton TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export FI_MODEL=$D/models/Qwen3.8-27B FI_TRACES=${FI_TRACES:-$D/v4/tmp_coord_gset_t005r_traces.json} FI_ROWS=""
cd $W/work; S=$1; shift; echo "[int] $(date) $S $*"; python -u $C/$S "$@" && echo "[int] $(date) done" || echo "[int] $(date) FAILED"
