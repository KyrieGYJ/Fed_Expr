#!/usr/bin/env bash

nohup bash run.sh 0.01 "resnet101" 200 64  10 "cuda:3" "mutual" "non-iid(1)" 2>&1 &

nohup bash run.sh 0.01 "resnet50" 200 64  10 "cuda:1" "mutual" "non-iid(1)" 2>&1 &

# 60 rounds 100 neighbor 30 stop
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:3" "local_and_mutual" "loss" "affinity" "non-iid" 60 100 999 "BCN_100n_aff" > log/BCN_100n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:3" "local" "loss" "flood" "non-iid" 60 100 30 "BCN_100c_local" > log/BCN_100c_local.log 2>&1 &

nohup bash run.sh 0.01 "resnet18" 200 64  10 "cuda:0" "mutual" "non-iid" 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:2" "local_and_mutual" "loss" "affinity" "non-iid" 60 100 999 "BCN-100c-local" > log/BCN-100c-local.log 2>&1 &

#bash run.sh 0.01 "BaseConvNet" 1 64  20 "cuda:3" "local_and_mutual" "loss" "affinity" "non-iid" 10 20 999 "BCN_100c_local"

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid" 60 20 999 "BCN_20n_aff" > log/BCN_20n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid" 60 50 999 "BCN_50n_aff" > log/BCN_50n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid" 60 100 999 "BCN_100n_aff" > log/BCN_100n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid" 60 20 30 "BCN_20n_aff_30stop" > log/BCN_20n_aff_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid" 60 50 30 "BCN_50n_aff_30stop" > log/BCN_50n_aff_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid" 60 100 30 "BCN_100n_aff_30stop" > log/BCN_100n_aff_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "flood" "non-iid" 60 20 999 "BCN_20n_flood" > log/BCN_20n_flood.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "flood" "non-iid" 60 50 999 "BCN_50n_flood" > log/BCN_50n_flood.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "flood" "non-iid" 60 100 999 "BCN_100n_flood" > log/BCN_100n_flood.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:1" "local_and_mutual" "loss" "flood" "non-iid" 60 20 30 "BCN_20n_flood_30stop" > log/BCN_20n_flood_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:1" "local_and_mutual" "loss" "flood" "non-iid" 60 50 30 "BCN_50n_flood_30stop" > log/BCN_50n_flood_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:1" "local_and_mutual" "loss" "flood" "non-iid" 60 100 30 "BCN_100n_flood_30stop" > log/BCN_100n_flood_30stop.log 2>&1 &


#rm BCN_20n_aff.log
#rm BCN_50n_aff.log
#rm BCN_100n_aff.log
#rm BCN_20n_aff_30stop.log
#rm BCN_50n_aff_30stop.log
#rm BCN_100n_aff_30stop.log

bash run.sh 0.01 "BaseConvNet" 1 64  10 "cuda:3" "oracle" "loss" "affinity" "non-iid" 1 20 999 "BCN_20n_aff"

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:1" "local" "loss" "affinity" "non-iid" 60 20 999 "BCN_100c_local" > log/BCN_100c_local.log 2>&1 &
