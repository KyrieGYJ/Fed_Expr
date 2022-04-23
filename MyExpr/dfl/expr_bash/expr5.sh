
# expr5 测试pretrain效果
# 设想pretrain后就不需要太多local train了

# dataset: cifar10 num_dist:2 local_train_epoch:1 lr:0.1

#  log/expr5/c60_local_latent2_nd2_pe50
bash run_expr.sh "TFConvNet" "cuda:2" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.1 50 "expr5"
# log/expr5/c60_wmi3_latent2_nd2_th0.1_pe50 382693
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.1 50 "expr5"
# log/expr5/c60_wmi3_latent2_nd2_th0.2_pe50 392580
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.2 50 "expr5"
# log/expr5/c60_wmi3_latent2_nd2_th0.5_pe50
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.5 50 "expr5"
# -------------------------------------- expr  -----------------------------------------