#!/usr/bin/env bash

#- client_num = 60
#- distribution= latent non-iid(distribution_num={2, 3, 4, 5, 10})
#- dataset：cifar10

# 这里150轮可能不太够，开到300试试
# -------------------- distribution_num = 2 ---------------------
# ours01 # log/expr2/c60_wmi3_latent2_nd2_th0.1_pe100
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.1 100 "expr2"
# ours03 # log/expr2/c60_wmi3_latent2_nd2_th0.3_pe100
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.3 100 "expr2"
# ours05 # log/expr2/c60_wmi3_latent2_nd2_th0.5_pe100
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.5 100 "expr2"
# local # log/expr2/c60_local_latent2_nd2_pe100
bash run_expr.sh "TFConvNet" "cuda:2" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.1 100 "expr2"


## local todo 跑之前记得打开wandb，切换到项目名，num_distribution也要调整
nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:2" "local" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_local_latent3" 0.2 3 > log_expr/c60_local_latent3.log 2>&1 &
#
#nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:2" "local" "loss" "affinity_topK" \
#"non-iid_latent2" 150 100 999 "c60_local_latent10" 0.2 10 > log_expr/c60_local_latent10.log 2>&1 &
#
#nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:2" "local" "loss" "affinity_topK" \
#"non-iid_latent2" 150 100 999 "c60_local_latent10" 0.2 10 > log_expr/c60_local_latent10.log 2>&1 &

#log_expr/c60_ours05_latent5.log

nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours02_latent2" 0.2 2 > log_expr/c60_ours02_latent2.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours02_latent3" 0.2 3 > log_expr/c60_ours02_latent3.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours02_latent4" 0.2 4 > log_expr/c60_ours02_latent4.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours02_latent5" 0.2 5 > log_expr/c60_ours02_latent5.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours02_latent10" 0.2 10 > log_expr/c60_ours02_latent10.log 2>&1 &

# todo 整理log，heatmap目录结构，按expr划分

# 300 rounds
# log/expr2/c60_wmi3_latent2_nd2_bk0.5_th0.1_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
300 0.5 2 0.1 100 20 "expr2"
# log/expr2/c60_local_latent2_nd2_bk0.5_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
300 0.5 2 0.1 100 20 "expr2"
# log/expr2/c60_wmi3_latent2_nd5_bk0.5_th0.1_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
300 0.5 5 0.1 100 20 "expr2"
# log/expr2/c60_local_latent2_nd5_bk0.5_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
300 0.5 5 0.1 100 20 "expr2"
# test
bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "test2" 0.5 2