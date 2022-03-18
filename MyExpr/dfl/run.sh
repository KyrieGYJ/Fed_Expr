#!/usr/bin/env bash

# todo 还有数据集选择
LR=$1
MODEL=$2
EPOCH=$3
BS=$4
N=$5
DEVICE=$6
TRAINER_STRATEGY=$7
TOPK_STRATEGY=$8
BROAD_STRATEGY=$9
DATA_DISTRIBUTION=${10}
COMM_ROUND=${11}
NEIGHBOR_UNDIRECTED=${12}
LOCAL_TRAIN_STOP_POINT=${13}
NAME=${14}


python3 main.py \
--lr $LR \
--model $MODEL \
--epoch $EPOCH \
--batch_size $BS \
--client_num_in_total $N \
--device $DEVICE \
--trainer_strategy $TRAINER_STRATEGY \
--topK_strategy $TOPK_STRATEGY \
--broadcaster_strategy $BROAD_STRATEGY \
--data_distribution $DATA_DISTRIBUTION \
--comm_round $COMM_ROUND \
--topology_neighbors_num_undirected $NEIGHBOR_UNDIRECTED \
--local_train_stop_point $LOCAL_TRAIN_STOP_POINT \
--name $NAME

