#!/usr/bin/env bash

nohup bash run.sh 0.01 "resnet101" 200 64  10 "cuda:3" "mutual" "non-iid(1)" 2>&1 &

nohup bash run.sh 0.01 "resnet50" 200 64  10 "cuda:1" "mutual" "non-iid(1)" 2>&1 &

# 60 rounds 100 neighbor 30 stop
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:3" "local_and_mutual" "loss" "affinity" "non-iid" 60 100 999 "BCN_100n_aff" > log/BCN_100n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:3" "local" "loss" "flood" "non-iid" 60 100 30 "BCN_100c_local" > log/BCN_100c_local.log 2>&1 &

nohup bash run.sh 0.01 "resnet18" 200 64  10 "cuda:0" "mutual" "non-iid" 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:2" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 60 100 999 "BCN-100c-local" > log/BCN-100c-local.log 2>&1 &

#bash run.sh 0.01 "BaseConvNet" 1 64  20 "cuda:3" "local_and_mutual" "loss" "affinity" "non-iid" 10 20 999 "BCN_100c_local"

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 60 20 999 "BCN_20n_aff" > log/BCN_20n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 60 50 999 "BCN_50n_aff" > log/BCN_50n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 60 100 999 "BCN_100n_aff" > log/BCN_100n_aff.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 60 20 30 "BCN_20n_aff_30stop" > log/BCN_20n_aff_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 60 50 30 "BCN_50n_aff_30stop" > log/BCN_50n_aff_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 60 100 30 "BCN_100n_aff_30stop" > log/BCN_100n_aff_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "flood" "non-iid_pathological" 60 20 999 "BCN_20n_flood" > log/BCN_20n_flood.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "flood" "non-iid_pathological" 60 50 999 "BCN_50n_flood" > log/BCN_50n_flood.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local_and_mutual" "loss" "flood" "non-iid_pathological" 60 100 999 "BCN_100n_flood" > log/BCN_100n_flood.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:1" "local_and_mutual" "loss" "flood" "non-iid_pathological" 60 20 30 "BCN_20n_flood_30stop" > log/BCN_20n_flood_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:1" "local_and_mutual" "loss" "flood" "non-iid_pathological" 60 50 30 "BCN_50n_flood_30stop" > log/BCN_50n_flood_30stop.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:1" "local_and_mutual" "loss" "flood" "non-iid_pathological" 60 100 30 "BCN_100n_flood_30stop" > log/BCN_100n_flood_30stop.log 2>&1 &


#rm BCN_20n_aff.log
#rm BCN_50n_aff.log
#rm BCN_100n_aff.log
#rm BCN_20n_aff_30stop.log
#rm BCN_50n_aff_30stop.log
#rm BCN_100n_aff_30stop.log

bash run.sh 0.01 "BaseConvNet" 1 64 15 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 20 15 999 "test2"

bash run.sh 0.01 "BaseConvNet" 1 64 15 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_latent2" 20 15 999 "test2"


bash run.sh 0.01 "BaseConvNet" 1 64 15 "cuda:3" "local_and_mutual" "loss" "affinity_baseline" "non-iid_latent2" 20 15 999 "test4"

bash run.sh 0.01 "BaseConvNet" 1 64 15 "cuda:3" "local_and_mutual" "loss" "affinity_cluster" "non-iid_latent2" 20 15 999 "test5"

# baseline
# local
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local" "loss" "affinity" "non-iid_pathological" 60 100 999 "BCN_local_pathological" > log/BCN_local_pathological.log 2>&1 &
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local" "loss" "affinity" "non-iid_latent" 60 100 999 "BCN_local_latent" > log/BCN_local_latent.log 2>&1 &
# oracle
nohup bash run.sh 0.01 "BaseConvNet" 1 64  10 "cuda:3" "oracle" "loss" "affinity" "non-iid_pathological" 1 20 999 "BCN_100c_oracle" > log/BCN_100c_oracle.log 2>&1 &
# fedAvg
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "fedavg" "loss" "affinity" "non-iid_pathological" 60 100 999 "BCN_fedavg_pathological" > log/BCN_fedavg_pathological.log 2>&1 &
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "fedavg" "loss" "affinity" "non-iid_latent" 60 100 999 "BCN_fedavg_latent" > log/BCN_fedavg_latent.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  10 "cuda:1" "fedavg" "loss" "affinity" "non-iid_latent" 60 20 999 "BCN_100c_fedavg" > log/BCN_100c_fedavg.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:3" "local" "loss" "affinity" "non-iid_latent" 60 100 30 "BCN_100c_local" > log/BCN_100c_local_latent.log 2>&1 &

# 3.25
nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "affinity" "non-iid_latent" 60 100 999 "BCN_100n_aff_latent" > log/BCN_100n_aff_latent.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "flood" "non-iid_latent" 80 100 999 "BCN_100n_flood_latent" > log/BCN_100n_flood_latent.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "affinity" "non-iid_pathological" 80 100 999 "BCN_100n_aff_pathological" > log/BCN_100n_aff_pathological.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "flood" "non-iid_pathological" 80 100 999 "BCN_100n_flood_pathological" > log/BCN_100n_flood_pathological.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local" "loss" "affinity" "non-iid_pathological" 60 100 999 "BCN_local_pathological" > log/BCN_local_pathological.log 2>&1 &
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local" "loss" "affinity" "non-iid_latent" 60 100 999 "BCN_local_latent" > log/BCN_local_latent.log 2>&1 &
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "fedavg" "loss" "affinity" "non-iid_pathological" 60 100 999 "BCN_fedavg_pathological" > log/BCN_fedavg_pathological.log 2>&1 &
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "fedavg" "loss" "affinity" "non-iid_latent" 60 100 999 "BCN_fedavg_latent" > log/BCN_fedavg_latent.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "no" "affinity" "non-iid_latent" 80 100 999 "aff_no_topk_latent" > log/aff_no_topk_latent.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "no" "flood" "non-iid_latent" 80 100 999 "flood_no_topk_latent" > log/flood_no_topk_latent.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "no" "affinity" "non-iid_pathological" 80 100 999 "aff_no_topk_pathological" > log/aff_no_topk_pathological.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "no" "flood" "non-iid_pathological" 80 100 999 "flood_no_topk_pathological" > log/flood_no_topk_pathological.log 2>&1 &

# 3.31
nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "local" "loss" "affinity" "non-iid_latent2" 60 100 999 "BCN_local_latent2" > log/BCN_local_pathological.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64  100 "cuda:0" "fedavg" "loss" "affinity" "non-iid_latent2" 60 100 999 "BCN_fedavg_latent2" > log/BCN_fedavg_pathological.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:0" "local_and_mutual" "loss" "affinity" "non-iid_latent2" 60 100 999 "BCN_100n_aff_latent2" > log/BCN_100n_aff_latent.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:0" "local_and_mutual" "loss" "flood" "non-iid_latent2" 80 100 999 "BCN_100n_flood_latent2" > log/BCN_100n_flood_pathological.log 2>&1 &

# 4.1
nohup bash run.sh 0.01 "BaseConvNet" 1 64 25 "cuda:0" "local_and_mutual" "loss" "affinity_cluster" "non-iid_latent2" 60 25 999 "BCN_25n_affC_latent2" > log/BCN_25n_affC_latent2.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 15 "cuda:0" "local_and_mutual" "loss" "affinity_cluster" "non-iid_latent2" 60 15 999 "BCN_15n_affC_latent2" > log/BCN_15n_affC_latent2.log 2>&1 &

nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:0" "local_and_mutual" "loss" "affinity_cluster" "non-iid_latent2" 60 100 999 "BCN_100n_affC_latent2" > log/BCN_100n_affC_latent2.log 2>&1 &
