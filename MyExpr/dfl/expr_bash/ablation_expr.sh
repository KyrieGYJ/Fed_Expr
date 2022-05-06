#!/usr/bin/env bash
# 消融实验（1），分别测试wmi和at的效果
# 可以继续追加在不同分布数量上时的效果
# todo 有可能要调大local train的epoch，减少communication round
# BK=0.5
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.5 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
1200 0.5 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.5 2 0.1 0 0 "ablation_expr"

# todo BK=0.4
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.4 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
1200 0.4 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.4 2 0.1 0 0 "ablation_expr"
# BK=0.3
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.3 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
1200 0.3 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.3 2 0.1 0 0 "ablation_expr"
# todo BK=0.2
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.2 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
1200 0.2 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.2 2 0.1 0 0 "ablation_expr"
# BK=0.1
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.1 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
1200 0.1 2 0.1 0 0 "ablation_expr"
bash run_expr.sh "TFConvNet" "cuda:3" "model_average" "random" 1 60 0.1 "non-iid_latent2" \
1200 0.1 2 0.1 0 0 "ablation_expr"
# 消融实验（2）探索r1，r2，chance的关系
# r1=(0.1 0.3 0.5)

arr=(1 2 3)
for i in ${arr[@]}
do

done