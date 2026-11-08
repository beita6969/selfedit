#!/bin/bash
# smoke.sh start_server CARD | stop_server CARD | train NAME CARDS -- bl_train args... | colo NAME CARDS -- bl_train args...
N=/home/bedicloud/sharestore2/zxc/selfedit; SHM=/dev/shm/selfedit-baseline; PY=$N/venv/bin/python; cd $N/baselines/code2
case $1 in
  start_server) c=$SHM/ctl/s$2; rm -rf $c; mkdir -p $c; bash run_cards.sh $2 rollout$2 70000 -- python -u bl_rollout.py --ctl $c ${3:-} ;;
  stop_server) c=$SHM/ctl/s$2; echo '{"op":"exit"}' > $c/.x; mv $c/.x $c/cmd_zzz.json ;;
  train) NAME=$2; CARDS=$3; shift 4; K=$(echo $CARDS | tr ',' '\n' | wc -l)
     bash run_cards.sh $CARDS $NAME ${BL_DEMAND:-70000} -- python -u -m torch.distributed.run --standalone --nproc_per_node=$K bl_train.py --name $NAME "$@" ;;
  colo) NAME=$2; CARDS=$3; shift 4
     bash run_cards.sh $CARDS $NAME ${BL_DEMAND:-75000} -- bash colo2.sh $NAME "$@" ;;
  colo3) NAME=$2; CARDS=$3; shift 4
     bash run_cards.sh $CARDS $NAME ${BL_DEMAND:-75000} -- bash colo3.sh $NAME "$@" ;;
esac
