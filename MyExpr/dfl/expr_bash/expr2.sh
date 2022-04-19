#!/usr/bin/env bash

#- client_num = 60
#- distribution= latent non-iid(distribution_num={2, 3, 4, 5, 10})
#- dataset：cifar10
#pwd
base_dir="."
dist_num=(2 3 4 5 10)
for i in ${dist_num[@]};
do
  if [ ! -d "${base_dir}/log_expr/c60_ours05_latent${i}.log" ]; then
    touch ${base_dir}/log_expr/c60_ours05_latent${i}.log
  fi
  nohup bash ${base_dir}/run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours05_latent${i}" 0.5 ${i} > ${base_dir}/log_expr/c60_ours05_latent${i}.log 2>&1 &
done

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

bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "test2" 0.2 2


# GPU3
59035