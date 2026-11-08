# Environment block of the main system's node jobs (code-hb/jobs/hb_n15.sh), with our own cache directory. Source it, then set CUDA_VISIBLE_DEVICES.
N=/home/bedicloud/sharestore2/zxc/selfedit; C=/dev/shm/selfedit-baseline/cache; CODE=$N/code-hb
export CUDA_DEVICE_ORDER=PCI_BUS_ID
. ${BL_VENV:-$N/venv}/bin/activate
unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset $(env | grep -o '^TRITON_[A-Z_]*')
mkdir -p $C/{hf,vllm,xdg,triton,inductor,nv,tmp}
export MPLBACKEND=Agg HF_HOME=$C/hf VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton \
       TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv TMPDIR=$C/tmp \
       OMP_NUM_THREADS=${BL_THREADS:-16} MKL_NUM_THREADS=${BL_THREADS:-16} HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
SP=$N/venv/lib/python3.12/site-packages
export CUDA_HOME=$SP/nvidia/cu13 PATH=$SP/nvidia/cu13/bin:$PATH FLASHINFER_WORKSPACE_BASE=$C NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
mkdir -p $C/cudalink && ln -sfn $SP/nvidia/cu13/lib/libcudart.so.13 $C/cudalink/libcudart.so
export LIBRARY_PATH=$C/cudalink:$SP/nvidia/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export PYTHONPATH=$N/baselines/code2:$CODE/engine:$CODE/mm${BL_FLA:+:$N/venv-baseline}
cd $N/baselines/code2
