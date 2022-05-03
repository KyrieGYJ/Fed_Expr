#!/usr/bin/env bash

#- client_num = 60
#- distribution= latent non-iid(distribution_num={2, 3, 4, 5, 10})
#- dataset：cifar10

# no pretrain
# todo 记录
# log/expr2/c60_local_latent2_nd2_bk0.5_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
400 0.5 2 0.1 0 20 "expr2"
# log/expr2/c60_local_latent2_nd5_bk0.5_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
400 0.5 5 0.1 0 20 "expr2"
# todo log/expr2/c60_wmi3_latent2_nd2_bk0.5_th0.1_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 20 "expr2"
# log/expr2/c60_wmi3_latent2_nd5_bk0.2_th0.1_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
400 0.2 5 0.1 0 20 "expr2"

# pretrain
# 400 rounds
# with malignant
# todo log/expr2/c60_local_latent2_nd2_bk0.5_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 100 20 "expr2"
# log/expr2/c60_local_latent2_nd5_bk0.2_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
400 0.2 5 0.1 100 20 "expr2"

# todo log/expr2/c60_wmi3_latent2_nd2_bk0.5_th0.1_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 100 20 "expr2"
# log/expr2/c60_wmi3_latent2_nd3_bk0.34_th0.1_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
400 0.34 3 0.1 100 20 "expr2"
# log/expr2/c60_wmi3_latent2_nd4_bk0.25_th0.1_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
400 0.25 4 0.1 100 20 "expr2"
# log/expr2/c60_wmi3_latent2_nd5_bk0.2_th0.1_pe100_mn20
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
400 0.2 5 0.1 100 20 "expr2"


# 4.27
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 100 20 "expr2" # log/expr2/c60_wmi3_random_latent2_nd2_bk0.5_th0.1
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 20 "expr2"
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 100 20 "expr2"

# 4.28
bash run_expr.sh "TFConvNet" "cuda:1" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:1" "model_average" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:1" "model_average" "random" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:1" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 20 "expr2"

# 带宽受限情况
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.1 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
500 0.1 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "model_average" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.1 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "model_average" "random" 1 60 0.1 "non-iid_latent2" \
500 0.1 2 0.1 0 0 "expr2"

# 选择合适的BK
# BK=0.2
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.2 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
500 0.2 2 0.1 0 0 "expr2"
# BK=0.3
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.3 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
500 0.3 2 0.1 0 0 "expr2"
# BK=0.4
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
500 0.4 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
500 0.4 2 0.1 0 0 "expr2"
# BK=0.5
bash run_expr.sh "TFConvNet" "cuda:0" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
1000 0.5 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:0" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
1000 0.5 2 0.1 0 0 "expr2"

