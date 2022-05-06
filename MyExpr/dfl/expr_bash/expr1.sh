#!/usr/bin/env bash

#- client_num={20, 100}
#- distribution={pathological non-iid (5 target)} # 这里dist_num设置为2，实际上无意义。
#- dataset=mnist, cifar10
#- K = 0.1, 0.3, 0.5
# client_num = 20
bash run_expr.sh "TFConvNet" "cuda:2" "local" "random" 1 20 0.1 "non-iid_pathological" \
500 0.2 2 0.1 0 0 "exp1"

bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
500 0.1 2 0.1 0 0 "exp1"

bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
500 0.3 2 0.1 0 0 "exp1"

bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
500 0.5 2 0.1 0 0 "exp1"

# client_num = 100
bash run_expr.sh "TFConvNet" "cuda:2" "local" "random" 1 100 0.1 "non-iid_pathological" \
500 0.2 2 0.1 0 0 "exp1"

bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
500 0.1 2 0.1 0 0 "exp1"

bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
500 0.3 2 0.1 0 0 "exp1"

bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
500 0.5 2 0.1 0 0 "exp1"

#MODEL=$1
#DEVICE=$2
#TRAINER_STRATEGY=$3
#BROAD_STRATEGY=$4
#EPOCH=$5
#N=$6
#LR=$7
#DATA_DISTRIBUTION=${8}
#COMM_ROUND=${9}
#BROADCAST_K=${10}
#NUM_DIST=${11}
#THRESHOLD=${12}
#PRETRAIN_EPOCH=${13}
#MALIGNANT_NUM=${14}
#EXPR_NAME=${15}