#!/usr/bin/env bash

#- client_num={20, 100}
#- distribution={pathological non-iid (5 target)} # 这里dist_num设置为2，实际上无意义。
#- dataset=mnist, cifar10
#- K = 0.2, 0.5

# cifar10
# local
# log/expr1/c20_local_pathl_nd2_pe100
bash run_expr.sh "TFConvNet" "cuda:2" "local" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
150 0.5 2 0.1 100 "expr1"
# log/expr1/c100_local_pathl_nd2_pe100
bash run_expr.sh "TFConvNet" "cuda:2" "local" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
150 0.5 2 0.1 100 "expr1"

# ours02
# todo
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
150 0.2 2 0.1 100 "expr1"
# todo
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
150 0.2 2 0.1 100 "expr1"

# ours05
# log/expr1/c20_wmi3_latent2_nd2_th0.1_pe100
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
150 0.5 2 0.1 100 "expr1"
# log/expr1/c100_wmi3_pathl_nd2_th0.1_pe100
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
150 0.5 2 0.1 100 "expr1"

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


# no pretrain
# log/expr1/c20_local_pathl_nd2
bash run_expr.sh "TFConvNet" "cuda:2" "local" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
150 0.5 2 0.1 0 "expr1"
# log/expr1/c100_local_pathl_nd2
bash run_expr.sh "TFConvNet" "cuda:2" "local" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
150 0.5 2 0.1 0 "expr1"
# log/expr1/c20_wmi3_pathl_nd2_th0.1
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 20 0.1 "non-iid_pathological" \
150 0.2 2 0.1 0 "expr1"
# log/expr1/c20_wmi3_pathl_nd2_th0.1
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 100 0.1 "non-iid_pathological" \
150 0.2 2 0.1 0 "expr1"