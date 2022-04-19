#!/usr/bin/env bash

#- client_num={20, 50, 100}
#- distribution={pathological non-iid (5 target)}
#- dataset=mnist, cifar10
#- K = 0.2, 0.5

# cifar10
# local
nohup bash run.sh 0.01 "TFConvNet" 1 64 20 "cuda:3" "local" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c20_local_path" 0.2 1 > log_expr/c20_local_path.log 2>&1 &
# best_acc:0.8382999897003174
nohup bash run.sh 0.01 "TFConvNet" 1 64 50 "cuda:3" "local" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c50_local_path" 0.2 1 > log_expr/c50_local_path.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 1 64 100 "cuda:3" "local" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c100_local_path" 0.2 1 > log_expr/c100_local_path.log 2>&1 &

# ours
# K=0.2
nohup bash run.sh 0.01 "TFConvNet" 1 64 20 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c20_ours02_path" 0.2 1 > log_expr/c20_ours02_path.log 2>&1 &
# best_acc:0.7928999662399292
nohup bash run.sh 0.01 "TFConvNet" 1 64 50 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c50_ours02_path" 0.2 1 > log_expr/c50_ours02_path.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c100_ours02_path" 0.2 1 > log_expr/c100_ours02_path.log 2>&1 &

# K=0.5
# best_acc:
nohup bash run.sh 0.01 "TFConvNet" 1 64 20 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c20_ours05_path" 0.5 1 > log_expr/c20_ours05_path.log 2>&1 &
# best_acc:0.8374999761581421
nohup bash run.sh 0.01 "TFConvNet" 1 64 50 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c50_ours05_path" 0.5 1 > log_expr/c50_ours05_path.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "c100_ours05_path" 0.5 1 > log_expr/c100_ours05_path.log 2>&1 &

# random + average

# FedAvg


#nohup bash run.sh > ../MyExpr/dfl/log_expr/c100_FedFomo_path.log 2>&1 &

bash run.sh 0.01 "TFConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological" 150 100 999 "test1" 0.5 1
