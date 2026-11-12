#!/bin/bash
# When an odd-row single worker finishes, re-point its card at the even short set (Vesta's half = even --shard 1/2; c7e runs it --reverse).
S=/workspace/fudan/mingdazhang/selfedit-signal; O=/data/fd/mingdazhang/selfedit/v4/signal; TR=/data/fd/mingdazhang/selfedit/v4/tmp_coord_gset_t005r_traces.json; MD=/data/fd/mingdazhang/selfedit/models/Qwen3.8-27B
PRE="unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=/data/fd/mingdazhang/selfedit/.hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/data/fd/mingdazhang/selfedit/.triton-sig; cd $O"
sh_of() { case $1 in 4) echo 1/4;; 5) echo 3/4;; 6) echo 1/2;; esac; }
DONE=""
while [ $(echo $DONE | wc -w) -lt 2 ]; do
  J=$(gpu jobs 2>/dev/null)
  for c in 4 5; do
    case " $DONE " in *" $c "*) continue;; esac
    st=$(echo "$J" | awk -v n="selfedit-v4s-f32c$c" '$1==n{print $2}')
    if [ -n "$st" ] && [ "$st" != running ]; then
      echo "$(date +%H:%M:%S) f32c$c $st -> even short shard $(sh_of $c)"
      gpu bg --name selfedit-v4s-f32c${c}e --gpus $c "$PRE; $S/venv/bin/python -u $S/nofla.py $S/score_worker_v32.py --traces $TR --rows mixed --row-shard 0/2 --shard $(sh_of $c) --f32 --max-total-tokens 6500 --out $O/scores_t005r_v3 --model $MD" | tail -1
      DONE="$DONE $c"
    fi
  done
  sleep 60
done
