#!/usr/bin/env bash

# cifar10
MODEL=$1
DEVICE=$2
N=$3
LR=$4
DATA_DISTRIBUTION=$5
NUM_DIST=$6
PRETRAIN_EPOCH=$7
EXPR_NAME=$8

NAME=c${N}_${DATA_DISTRIBUTION}_nd${NUM_DIST}_pe${PRETRAIN_EPOCH}
echo "预训练:${NAME}"

python3 pretrain.py\
  --lr $LR \
  --model $MODEL \
  --client_num_in_total $N \
  --num_distributions $NUM_DIST \
  --device $DEVICE \
  --data_distribution $DATA_DISTRIBUTION \
  --name $NAME \
  --project_name $EXPR_NAME \
  --pretrain_epoch $PRETRAIN_EPOCH