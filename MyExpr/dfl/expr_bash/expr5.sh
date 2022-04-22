
# 设想pretrain后就不需要太多local train了
# expr5 测试pretrain效果

bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 2 1 "test"

# ---------------------------------------- pretrain  ------------------------------------------------
# ------------------------- client_num:60, pretrain_epoch:{20, 50, 100} ------------------------------
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 2 20 "expr5" \
> log/pretrain/c60_non-iid_latent2_nd2_pe20.log 2>&1 & # 30min
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 2 50 "expr5" \
> log/pretrain/c60_non-iid_latent2_nd2_pe50.log 2>&1 & # 70min
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 2 100 "expr5" \
> log/pretrain/c60_non-iid_latent2_nd2_pe100.log 2>&1 &
# -------------------------------------- pretrain  -----------------------------------------


# -------------------------------------- expr  -----------------------------------------
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