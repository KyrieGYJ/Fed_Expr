#!/usr/bin/env bash

# cifar10
MODEL=$1
DEVICE=$2
TRAINER_STRATEGY=$3
BROAD_STRATEGY=$4
EPOCH=$5
N=$6
LR=$7
DATA_DISTRIBUTION=${8}
COMM_ROUND=${9}
BROADCAST_K=${10}
NUM_DIST=${11}
THRESHOLD=${12}
PRETRAIN_EPOCH=${13}
NAME=${14}
EXPR_NAME=${15}

echo "当前试验："$NAME

python3 main.py \
  --lr $LR \
  --model $MODEL \
  --epoch $EPOCH \
  --client_num_in_total $N \
  --broadcast_K $BROADCAST_K \
  --num_distributions $NUM_DIST \
  --device $DEVICE \
  --trainer_strategy $TRAINER_STRATEGY \
  --broadcaster_strategy $BROAD_STRATEGY \
  --data_distribution $DATA_DISTRIBUTION \
  --comm_round $COMM_ROUND \
  --aggregate_threshold $THRESHOLD \
  --name $NAME \
  --project_name $EXPR_NAME \
  --pretrain_epoch $PRETRAIN_EPOCH

## todo 包装成多个函数
#LR=$1
#MODEL=$2
#EPOCH=$3
##BS=$4 #
#N=$5
#DEVICE=$6
#TRAINER_STRATEGY=$7
##TOPK_STRATEGY=$8 #
#BROAD_STRATEGY=$9
#DATA_DISTRIBUTION=${10}
#COMM_ROUND=${11}
##NEIGHBOR_UNDIRECTED=${12} #
##LOCAL_TRAIN_STOP_POINT=${13}
#NAME=${14}
#BK=${15}
#NUM_DIST=${16}
#
#
#python3 main.py \
#--lr $LR \
#--model $MODEL \
#--epoch $EPOCH \
#--batch_size $BS \
#--client_num_in_total $N \
#--broadcast_K $BK \
#--num_distributions $NUM_DIST \
#--device $DEVICE \
#--trainer_strategy $TRAINER_STRATEGY \
#--topK_strategy $TOPK_STRATEGY \
#--broadcaster_strategy $BROAD_STRATEGY \
#--data_distribution $DATA_DISTRIBUTION \
#--comm_round $COMM_ROUND \
#--topology_neighbors_num_undirected $NEIGHBOR_UNDIRECTED \
#--local_train_stop_point $LOCAL_TRAIN_STOP_POINT \
#--name $NAME

