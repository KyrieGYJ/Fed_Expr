#!/usr/bin/env bash

# baseline
bash run_expr.sh "TFConvNet" "cuda:0" "fedavg" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"
bash run_expr.sh "TFConvNet" "cuda:0" "fedprox" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"
bash run_expr.sh "TFConvNet" "cuda:0" "pfedme" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"
bash run_expr.sh "TFConvNet" "cuda:0" "apfl" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"


bash run_expr.sh "TFConvNet" "cuda:0" "weighted_model_interpolation3" "affinity_topK" 1 30 0.1 "non-iid_latent2" \
50 0.1 5 0.1 0 0 "test"

bash run_expr.sh "TFConvNet" "cuda:0" "weighted_model_interpolation3" "affinity_topK" 1 30 0.1 "non-iid_latent2" \
5 0.1 5 0.1 0 0 "test2"

bash run_expr.sh "TFConvNet" "cuda:0" "apfl" "affinity_topK" 1 10 0.1 "non-iid_latent2" \
3 0.1 5 0.1 0 0 "test2"