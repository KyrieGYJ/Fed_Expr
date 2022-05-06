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
bash run_expr.sh "TFConvNet" "cuda:0" "fedem" "random" 1 10 0.1 "non-iid_pathological" \
10 0.2 2 0.1 0 0 "test"