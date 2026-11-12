#!/bin/bash
# usage: to_w.sh CARD...  drain the (released) E server on each card, stop it, start a W worker W<card> (util .92, 96 seqs), mark the port released/W.
ED=/workspace/fudan/mingdazhang/selfedit-v4/engine_v4; V4=/data/fd/mingdazhang/selfedit/v4
cd ~/selfedit/code-v4/engine_v4
TAG=t$(date +%H%M)
for c in "$@"; do
  echo "{\"820$c\": {\"state\": \"released\"}}" > spec_rel_$c.json
  gpu put spec_rel_$c.json $V4/ckpt/spec_rel_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_rel_$c.json" | tail -1
done
for c in "$@"; do for i in $(seq 1 80); do n=$(gpu run "curl -s 127.0.0.1:820$c/metrics | grep -E '^vllm:num_requests_(running|waiting)' | awk '{s+=\$2} END{print s+0}'" 2>/dev/null | tr -d '\r' | tail -1); echo "$(date +%H:%M:%S) card $c in flight $n"; [ "$n" = "0" ] && break; sleep 15; done; done
for c in "$@"; do
  for j in $(gpu jobs | awk -v c=$c '$1 ~ "^selfedit-v4e-[sv]"c && $2=="running" {print $1}'); do gpu stop $j | tail -1; done
  for i in $(seq 1 40); do u=$(gpu run "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c" 2>/dev/null | tr -d '\r' | tail -1); echo "$(date +%H:%M:%S) card $c memory used ${u} MiB"; [ -n "$u" ] && [ "$u" -lt 2000 ] 2>/dev/null && break; sleep 10; done
  gpu bg --name selfedit-v4e-w${c}x$TAG --gpus $c "SE=/workspace/fudan/mingdazhang/selfedit; D=/data/fd/mingdazhang/selfedit; . \$SE/venv/bin/activate; unset LD_LIBRARY_PATH TORCH_ALLOW_TF32_CUBLAS_OVERRIDE; unset \$(env | grep -o '^TRITON_[A-Z_]*'); export MPLBACKEND=Agg HF_HOME=\$D/.hf VLLM_CACHE_ROOT=\$D/.vllm-cache; cd \$D/v4 && python -u $ED/w_worker.py --name W$c --util 0.92 --max-num-seqs 96" | tail -1
  echo "{\"820$c\": {\"state\": \"released\", \"note\": \"card is W worker W$c (selfedit-v4e-w${c}x$TAG)\"}}" > spec_w_$c.json
  gpu put spec_w_$c.json $V4/ckpt/spec_w_$c.json >/dev/null && gpu run "python3 $ED/eng_set.py $V4/ckpt/spec_w_$c.json" | tail -1
  echo "$(date +%H:%M:%S) card $c: W$c starting"
done
