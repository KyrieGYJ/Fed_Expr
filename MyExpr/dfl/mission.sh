#!/usr/bin/env bash

nohup bash run.sh 0.01 "resnet101" 200 64  10 "cuda:3" "mutual" "non-iid(1)" 2>&1 &

nohup bash run.sh 0.01 "resnet50" 200 64  10 "cuda:1" "mutual" "non-iid(1)" 2>&1 &

# 60 rounds 100 neighbor 30 stop
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:3" "local_and_mutual" "loss" "affinity" "non-iid" 60 100 999 "BCN_100n_aff" > log/BCN_100n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:3" "local_and_mutual" "loss" "flood" "non-iid" 60 100 30 "BCN_100n_flo_30stop" > log/BCN_100n_flo_30stop.log 2>&1 &

nohup bash run.sh 0.01 "resnet18" 200 64  10 "cuda:0" "mutual" "non-iid(1)" 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:2" "local_and_mutual" "loss" "affinity" "non-iid" 60 100 999 "BCN-100c-local" > log/BCN-100c-local.log 2>&1 &
