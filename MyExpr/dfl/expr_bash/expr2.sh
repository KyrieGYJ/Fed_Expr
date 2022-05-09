#!/usr/bin/env bash

#- client_num = (20, 100)
#- distribution= latent non-iid(distribution_num={2, 3, 4, 5, 10})
#- BK={0.1, 0.3, 0.5}
#- dataset：cifar10

bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 3 100 0.1 "non-iid_latent2" \
500 0.1 2 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 3 100 0.1 "non-iid_latent2" \
500 0.1 3 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 3 100 0.1 "non-iid_latent2" \
500 0.1 4 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 3 100 0.1 "non-iid_latent2" \
500 0.1 5 0.1 0 0 "expr2"

# todo
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 3 100 0.1 "non-iid_latent2" \
500 0.1 10 0.1 0 0 "expr2"


bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 3 100 0.1 "non-iid_latent2" \
500 0.3 2 0.1 0 0 "exp2"
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 3 100 0.1 "non-iid_latent2" \
500 0.5 2 0.1 0 0 "exp2"



# todo 对比 local batch变化
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 5 100 0.1 "non-iid_latent2" \
200 0.1 5 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 1 100 0.1 "non-iid_latent2" \
1500 0.1 5 0.1 0 0 "expr2_test"

# baseline
bash run_expr.sh "TFConvNet" "cuda:2" "fedavg" "affinity_topK" 5 100 0.1 "non-iid_latent2" \
200 0.5 5 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "fedprox" "affinity_topK" 5 100 0.1 "non-iid_latent2" \
200 0.5 5 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "pfedme" "affinity_topK" 5 100 0.1 "non-iid_latent2" \
200 0.5 5 0.1 0 0 "expr2"
bash run_expr.sh "TFConvNet" "cuda:2" "apfl" "affinity_topK" 5 100 0.1 "non-iid_latent2" \
200 0.5 5 0.1 0 0 "expr2"

bash run_expr.sh "TFConvNet" "cuda:0" "fedavg" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"
bash run_expr.sh "TFConvNet" "cuda:0" "fedprox" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"
bash run_expr.sh "TFConvNet" "cuda:0" "pfedme" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"
bash run_expr.sh "TFConvNet" "cuda:0" "apfl" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"