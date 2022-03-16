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
DATA_DISTRIBUTION=$9
NAME=${MODEL}-lr${LR}-bc${BS}-ts${TRAINER_STRATEGY}-${DATA_DISTRIBUTION}

echo $NAME

python3 main.py \
--lr $LR \
--model $MODEL \
--epoch $EPOCH \
--batch_size $BS \
--client_num_in_total $N \
--device $DEVICE \
--trainer_strategy $TRAINER_STRATEGY \
--topK_strategy $TOPK_STRATEGY \
--data_distribution $DATA_DISTRIBUTION \
--name $NAME

