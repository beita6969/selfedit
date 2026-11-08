#!/bin/bash
# launch128.sh stop | arm GRADE_MM_SHA HBOOST_SHA BUDGET [EFFORT]   (owner 11-08 plan; data in baselines/data128_v3 (pool v3), code in code2)
#   BUDGET = 8192 | 16384 | 32768 and EFFORT (default medium): the main system's answer budget and effort, used for GRPO rollouts, every evaluation,
#   SFT traces and the final IID (bl_final_eval.py --budget). Every engine reply is checked to have used exactly these.
#   stop: stops grpo1 and sft1 cleanly (their own launcher trees only), my sync_guard and my card-5 engine (old code); their files and results stay.
#   arm:  waits, then launches. Order: (1) the grader that code2 imports must have these sha values and ck_grade_all.py must pass
#         (0 exceptions, also none caught by grade_rec; every gold right); (2) no grpo1/sft1 trainer left; (3) space check passes (after the
#         owner's cleanup); then grpo2 (cards 2,4 + card-5 engine); once grpo2 has evaluated stock, a second space check, then sft2 (6,1: traces, then training).
# Space model: one new run's RAM footprint P is measured, not assumed. 'stop' records RAM available with grpo1 and sft1 running (A_before).
# After the stop and the owner's cleanup, A_now - A_before is what the two old runs held. A new run keeps no best copy (-52 GB each) and sft2
# will hold a rollout_model that sft1 lacked (+52 GB), so P = (A_now - A_before - 52) / 2. arm waits for the cleanup before measuring.
set -u
N=/home/bedicloud/sharestore2/zxc/selfedit; SHM=/dev/shm/selfedit-baseline; D=$N/baselines/data128_v3; export BL_DATA=$D BL_POOL=$N/baselines/xfer128_v3/iid_paper128_v3.jsonl; C2=$N/baselines/code2; LOG=$N/baselines/logs; cd $C2
say() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
ram() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
shm() { df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9; }
old_running() { pgrep -f "bl_train.py --name (grpo1|sft1)( |$)" > /dev/null; }
kids() { for k in $(ps -o pid= --ppid $1); do echo $k; kids $k; done; }
case "${1:-}" in
stop)
  for NAME in grpo1 sft1; do
    C=$(ps -eo pid,args | grep -E "^ *[0-9]+ bash colo[23]?\.sh $NAME " | awk '{print $1}')
    [ -z "$C" ] && { say "$NAME: no launcher (already ended)"; continue; }
    [ $(echo $C | wc -w) = 1 ] || { say "no unique launcher for $NAME: $C"; exit 1; }
    K=$(kids $C); say "stopping $NAME launcher $C and $(echo $K | wc -w) children"; kill $C; kill $K 2>/dev/null
  done
  G=$(ps -eo pid,args | awk '$2=="python3" && $3=="-I" && $4=="sync_guard.py" {print $1}'); [ -n "$G" ] && kill $G && say "sync_guard $G stopped"
  if [ -d $SHM/ctl/s5 ] && pgrep -f "bl_rollout.py --ctl $SHM/ctl/s5 " > /dev/null; then bash smoke.sh stop_server 5; say "card-5 engine asked to exit"; fi
  echo $(ram) > $SHM/ram_before_stop.txt; say "RAM available before the stop: $(cat $SHM/ram_before_stop.txt) GB"
  for t in $(seq 30); do old_running || break; sleep 2; done
  old_running && { K=$(pgrep -f "bl_train.py --name (grpo1|sft1)( |$)"); say "still running after 60 s, kill -9: $K"; kill -9 $K; }
  for NAME in grpo1 sft1; do S=$SHM/$NAME; if ls $S/.sync_* > /dev/null 2>&1; then X=$S/stale_sync_$(date +%H%M%S); mkdir -p $X; mv $S/.sync_* $X/; fi; done
  say "stopped; RAM available $(ram) GB, shm free $(shm) GB"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo ;;
arm)
  WANT_G=${2:?grade_mm sha}; WANT_H=${3:?hboost sha}; BUDGET=${4:?budget 8192, 16384 or 32768}; EFFORT=${5:-medium}
  case $BUDGET in 8192|16384|32768) ;; *) say "budget $BUDGET not allowed"; exit 1;; esac
  RUNARGS="--budget $BUDGET --effort $EFFORT --cap-strict 1";
  # engine settings from tests32k (11-08 16:4x CST): prefix cache + 16k prefill chunk + 192 seqs (240 measured 2x tok/s vs 128 without; vLLM refuses >245 at util 0.85)
  # trainers use the fla kernels (BL_FLA=1, $N/venv-baseline) and 28 threads, as grpo1 and sft1 did; without fla a 46k sample runs out of card memory
  # BL_ARM_SEQS / BL_ARM_ENG override them, e.g. the main run's regime: BL_ARM_SEQS=64 BL_ARM_ENG="" (64 seqs, 8k chunk, no prefix cache)
  SEQS=${BL_ARM_SEQS:-192}; ENG=${BL_ARM_ENG-"--prefix-cache 1 --max-batched 16384"}; say "engines: $SEQS seqs, extra [$ENG]"; say "budget $BUDGET, effort $EFFORT"; [ -s $SHM/ram_before_stop.txt ] || { say "run 'stop' first"; exit 1; }
  for f in train.jsonl val.jsonl paper_select.node.jsonl iid_eval.node.jsonl counts.json; do [ -s $D/$f ] || { say "missing $D/$f"; exit 1; }; done
  . ./env.sh > /dev/null 2>&1
  out=$(CUDA_VISIBLE_DEVICES="" nice python ck_grade_all.py 2>/dev/null | head -1)
  say "grader check: ${out:0:400}"
  echo "$out" | python3 -I -c "import json,sys; x=json.loads(sys.stdin.read()); g=x['grader']; c=x['corrupted']; ok = g['grade_mm']==sys.argv[1][:16] and g['hboost']==sys.argv[2][:16] and g['pool']=='ff0d465bcd0eac58' and g['grade_mm_loaded'].endswith('code2/hbgrade/mm/grade_mm.py') and not x['exceptions'] and x['gold_wrong']==0 and x['rows']==1536 and x['stored_gold_differs_from_pool']==0 and c['differ']==0 and c['gold_right_after_corruption']==1536; sys.exit(0 if ok else 1)" $WANT_G $WANT_H \
    || { say "grader check FAILED or sha differs (want grade_mm $WANT_G hboost $WANT_H): not launching"; exit 1; }
  while old_running; do say "grpo1/sft1 trainer still running; waiting"; sleep 60; done
  until [ ! -e $SHM/grpo1/ckpt ] && [ ! -e $SHM/grpo1/best_model ] && [ ! -e $SHM/grpo1/rollout_model ] && [ ! -e $SHM/sft1/ckpt ] && [ ! -e $SHM/sft1/best_model ]; do
    say "waiting for the owner's cleanup (owner_cleanup_shm.sh): RAM $(ram) GB, shm $(shm) GB"; sleep 60; done
  sleep 30; A0=$(cat $SHM/ram_before_stop.txt); P=$(( ($(ram) - A0 - 52) / 2 )); say "cleanup seen: RAM $(ram) GB (before stop $A0), one new run P = $P GB"
  [ $P -lt 200 ] && { say "P = $P GB looks too small to be real: not launching; report"; exit 2; }
  need1=$(( P + 102 + 100 ))
  until [ $(ram) -ge $need1 ] && [ $(shm) -ge $((154 + 102 + 100)) ]; do say "waiting for room for grpo2: RAM $(ram) GB, shm $(shm) GB (need RAM $need1, shm 356)"; sleep 60; done
  say "room for grpo2: RAM $(ram) GB, shm $(shm) GB"
  pgrep -f "bl_rollout.py --ctl $SHM/ctl/s5 " > /dev/null && { say "an engine on ctl/s5 is still running: not launching"; exit 1; }
  for t in $(seq 60); do u=$(nvidia-smi -i 5 --query-gpu=memory.used --format=csv,noheader,nounits); [ $u -lt 5000 ] && break; sleep 5; done
  nohup bash smoke.sh start_server 5 "--num-seqs $SEQS --sleep-mode 1 --seed 106 $ENG" > $LOG/server5-code2.log 2>&1 &
  for t in $(seq 120); do [ -e $SHM/ctl/s5/ready ] && break; sleep 5; done; [ -e $SHM/ctl/s5/ready ] || { say "card-5 engine not ready after 10 min"; exit 1; }
  say "card-5 engine (code2) ready"
  cat > $N/baselines/overrides/grpo2.json <<J
{"max_hours": 21.5, "patience": 0}
J
  BL_FLA=1 BL_THREADS=28 BL_EXTRA_CTL=$SHM/ctl/s5 BL_SEQS=$SEQS BL_ENGINE_EXTRA="$ENG" nohup bash smoke.sh colo3 grpo2 2,4 -- --method grpo --q 64 --g 8 --mb 8 --iters 100000 --lr 2e-6 --warm 20 \
    --actor-offload 1 --critic-offload 1 --max-tok 36000 --val-every 6 --ckpt-every 6 --ckpt-adam 0 --patience 0 --keep-best 0 --final-iid 0 --val-first 1 --max-hours 21.5 $RUNARGS \
    > $LOG/launch-grpo2.log 2>&1 &
  say "grpo2 launched (pid $!)"
  for t in $(seq 120); do L=$(ls -t $LOG/grpo2-[0-9]* 2>/dev/null | head -1); [ -n "$L" ] && grep -qE '"event": "(val|stop|end)"|Traceback' $L && break; sleep 30; done
  L=$(ls -t $LOG/grpo2-[0-9]* 2>/dev/null | head -1); say "grpo2 log $L: $(grep -E '"event": "(val|stop|end)"|Traceback' $L | head -2 | cut -c1-300)"
  grep -q Traceback $L && { say "grpo2 failed: sft2 not started"; exit 1; }
  grep '"event": "start"' $L | grep -q "code2/hbgrade/mm/grade_mm.py" && grep '"event": "start"' $L | grep -q "\"budget\": $BUDGET" || { say "grpo2 start event lacks the code2 grader or budget $BUDGET: sft2 not started"; exit 1; }
  need2=$(( P + 102 + 102 + 100 ))      # sft2's footprint, grpo2's checkpoint not yet written, one checkpoint transient, floor
  for t in $(seq 120); do [ $(ram) -ge $need2 ] && [ $(shm) -ge $((154 + 102 + 102 + 100)) ] && break; say "waiting for room for sft2: RAM $(ram) GB, shm $(shm) GB (need RAM $need2)"; sleep 60; done
  [ $(ram) -ge $need2 ] || { say "no room for sft2 after 2 h: not started; report"; exit 2; }
  cat > $N/baselines/overrides/sft2.json <<J
{"max_hours": 21.5, "patience": 0}
J
  mkdir -p $SHM/sft2
  BL_PRE="python -u bl_sft_data.py --ctl \$CTL --out $SHM/sft2/traces.jsonl --per-row 4 --passes 8 --hours 4 --chunk 384 --budget $BUDGET --effort $EFFORT" BL_FLA=1 BL_THREADS=28 BL_SEQS=$SEQS BL_ENGINE_EXTRA="$ENG" nohup bash smoke.sh colo sft2 6,1 -- --method sft \
    --traces $SHM/sft2/traces.jsonl --batch 32 --iters 480 --lr 1e-6 --warm -1 --actor-offload 1 --critic-offload 1 --max-tok 36000 --val-every 70 --ckpt-every 70 \
    --ckpt-adam 0 --patience 0 --keep-best 0 --final-iid 0 --val-first 0 --max-hours 21.5 $RUNARGS > $LOG/launch-sft2.log 2>&1 &
  say "sft2 launched (pid $!); RAM $(ram) GB, shm $(shm) GB" ;;
*) echo "usage: launch128.sh stop | arm GRADE_MM_SHA HBOOST_SHA BUDGET [EFFORT]"; exit 2 ;;
esac
