#!/usr/bin/env bash

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
MALIGNANT_NUM=${14}
EXPR_NAME=${15}

NAME=c${N}

# 缩短名称
if [[ "$TRAINER_STRATEGY" =~ "weighted_model_interpolation" ]]
then
  NAME=${NAME}_wmi${TRAINER_STRATEGY:0-1}
else
  NAME=${NAME}_${TRAINER_STRATEGY}
fi

if [[ "$DATA_DISTRIBUTION" =~ "path" ]]
then
  NAME=${NAME}_path${DATA_DISTRIBUTION:0-1}
elif [[ "$DATA_DISTRIBUTION" =~ "latent" ]]
then
  NAME=${NAME}_latent${DATA_DISTRIBUTION:0-1}
fi
NAME=${NAME}_nd${NUM_DIST}
NAME=${NAME}_bk${BROADCAST_K}
if [[ "$TRAINER_STRATEGY" =~  "weighted" ]]
then
NAME=${NAME}_th${THRESHOLD}
fi

if [ $PRETRAIN_EPOCH -gt 0 ]
then
  NAME=${NAME}_pe${PRETRAIN_EPOCH}
fi

if [ $MALIGNANT_NUM -gt 0 ]
then
  NAME=${NAME}_mn${MALIGNANT_NUM}
fi

if [[ $EXPR_NAME == "" || $EXPR_NAME == "bash" ]]
then
  echo "expr_name is '', use default turn of wandb"
  EXPR_NAME="default"
fi

# 不存在文件夹则自动创建
if [ ! -d "log/${EXPR_NAME}" ]
then
  echo "日志文件夹不存在: log/${EXPR_NAME} ，自动创建..."
  mkdir log/${EXPR_NAME}
fi

# 不存在热力图文件夹则自动创建
if [ ! -d "heatmap/${EXPR_NAME}" ]
then
  echo "热力图文件夹不存在: heatmap/${EXPR_NAME} ，自动创建..."
  mkdir heatmap/${EXPR_NAME}
fi

# todo 需要重复考虑末尾加上日期
# 日志文件存在则终止运行，给出提示
if [ -e "log/${EXPR_NAME}/${NAME}" ]
then
  echo "日志文件已存在，请手动检查: log/${EXPR_NAME}/${NAME} ."
else
  echo "启动试验，日志位置: log/${EXPR_NAME}/${NAME} ."
  nohup bash run.sh $MODEL $DEVICE $TRAINER_STRATEGY $BROAD_STRATEGY $EPOCH  $N $LR \
        $DATA_DISTRIBUTION $COMM_ROUND $BROADCAST_K $NUM_DIST $THRESHOLD ${PRETRAIN_EPOCH} \
        ${MALIGNANT_NUM} ${NAME} ${EXPR_NAME} > log/${EXPR_NAME}/${NAME} 2>&1 &
fi
