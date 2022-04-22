#!/usr/bin/env bash

bash run.sh 0.01 "FedAvgCNN" 1 64 100 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
 "non-iid_pathological2" 50 100 999 "test_mnist" 1

bash run.sh 0.01 "TFConvNet" 1 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "test1" 0.2 4

# test
# log/test/c60_local_non-iid_latent2_nd2_pe1
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.6 1 "test"
# 224595
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
150 0.5 2 0.1 1 "test"

bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 5 60 0.1 "non-iid_pathological2" \
150 0.5 5 0.1 0 "test"

bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 1 60 0.1 "non-iid_latent2" \
10 0.5 2 0.1 50 "test"

# 154139 192621 444189 224595 382693 392580 170268 447293 1625


## over different data setting
#
## baseline
## fedavg
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "fedavg" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "fedavg_path2" 1 > log/fedavg_pathological2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "fedavg" "loss" "affinity_topK" "iid" 100 100 999 "fedavg_iid" 1 > log/fedavg_iid.log 2>&1 &
## oracle
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "oracle_distribution" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "oracle_path2" -1 > log/oracle_dist_pathological2.log 2>&1 &
## local
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological" 100 100 999 "local_path2" 1 > log/local_pathological2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local" "loss" "affinity" "iid" 100 100 999 "local_iid" 1 > log/local_iid.log 2>&1 &
#
## topK
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "topk_0.5" 0.5 > log/mutual_test/topk_05.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "topk_1" 1 > log/mutual_test/topk_04.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 20 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "topk_0.2" 0.2 > log/mutual_test/topk_02.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "topk_-1" -1 > log/topk_-1.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "topk_-1" 0.5 > log/mta_dp.log 2>&1 &
#
#
## total = 100, client_per_dist = 20
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 20 "cuda:1" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "topk_0.5" 0.5 > log/non_iid_20/topk_05.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 20 "cuda:1" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "topk_0.2" 0.2 > log/non_iid_20/topk_02.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 20 "cuda:1" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "topk_0.1" 0.1 > log/non_iid_20/topk_01.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 20 "cuda:2" "fedavg" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "fedavg" 0.1 > log/non_iid_20/fedavg.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 2 64 20 "cuda:2" "local" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "local2" 1 > log/local2.log 2>&1 &
#
#
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 20 "cuda:1" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "topk_0.05" 0.05 > log/non_iid_20/topk_005.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 20 "cuda:1" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "topk_0.02" 0.02 > log/non_iid_20/topk_002.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "fedavg" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "fedavg_path2" 1 > log/fedavg_pathl2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "oracle_distribution" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "oracle_path2" -1 > log/oracle_dist_pathological2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "local_path2" 1 > log/local_path2.log 2>&1 &
#
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:1" "local_and_mutual" "loss" "random" "non-iid_pathological2" 50 100 999 "random05" 0.5 > log/random05.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:2" "fedavg" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "fedavg_path2" 1 > log/dfl_20client_iid/fedavg_path2.log 2>&1 &
#
#
#bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "testtest" 1
#
#bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:2" "fedavg" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "fedavg_path2" 1
#
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "fedavg" "loss" "affinity_topK" "non-iid_pathological" 100 100 999 "c10_path" 1 > log/fedavg/c10_path.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 50 "cuda:3" "fedavg" "loss" "affinity_topK" "non-iid_pathological" 100 100 999 "c20_path" 1 > log/fedavg/c20_path.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "fedavg" "loss" "affinity_topK" "non-iid_pathological" 100 100 999 "c100_path" 1 > log/fedavg/c100_path.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "fedavg" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c10_path2" 1 > log/fedavg/c10_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "fedavg" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c100_path2" 1 > log/fedavg/c100_path2.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "fedavg" "loss" "affinity_topK" "iid" 100 100 999 "c10_iid" 1 > log/fedavg/c10_iid.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "fedavg" "loss" "affinity_topK" "iid" 100 100 999 "c100_iid" 1 > log/fedavg/c100_iid.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:2" "fedavg" "loss" "affinity_topK" "non-iid_latent" 100 100 999 "c10_latent" 1 > log/fedavg/c10_latent.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:2" "fedavg" "loss" "affinity_topK" "non-iid_latent" 100 100 999 "c100_latent" 1 > log/fedavg/c100_latent.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c100_new_path2" 1 > log/dfl_weight/c100_new_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c10_new_path2" 1 > log/dfl_weight/c10_new_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 2 64 100 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c100_local2_path2" 1 > log/dfl_weight/c100_local2_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 2 64 10 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c10_local2_path2" 1 > log/dfl_weight/c10_local2_path2.log 2>&1 &
#
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c10_new_mutual_path2" 1 > log/dfl_weight/c10_new_mutual_path2.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "local_and_mutual" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c10_new_m2_path2" 1 > log/dfl_weight/c10_new_m2_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 3 64 10 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c10_local3_path2" 1 > log/dfl_weight/c10_local3_path2.log 2>&1 &
#
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c10_local1_path2" 1 > log/dfl_weight/c10_local1_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c100_local1_path2" 1 > log/dfl_weight/c10_local1_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c10_wmi_path2" 1 > log/dfl_weight/c10_wmi_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "c100_wmi_path2" 1 > log/dfl_weight/c100_wmi_path2.log 2>&1 &
#
#
#bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "testtest" 1
#bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation2" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "testtest" 1
#bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" "non-iid_pathological2" 50 100 999 "test_mnist" 1
#bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation5" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "testtest" 1
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c10_wmi1_path2" 1 > log/dfl_mwi/c10_wmi1_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c100_wmi1_path2" 1 > log/dfl_mwi/c100_wmi1_path2.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation2" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c10_wmi2_path2" 1 > log/dfl_mwi/c10_wmi2_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation2" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c100_wmi2_path2" 1 > log/dfl_mwi/c100_wmi2_path2.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c10_wmi3_path2" 1 > log/dfl_mwi/c10_wmi3_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c100_wmi3_path2" 1 > log/dfl_mwi/c100_wmi3_path2.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation4" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c10_wmi4_path2" 1 > log/dfl_mwi/c10_wmi4_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation4" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c100_wmi4_path2" 1 > log/dfl_mwi/c100_wmi4_path2.log 2>&1 &
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "weighted_model_interpolation5" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c10_wmi5_path2" 1 > log/dfl_mwi/c10_wmi5_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "weighted_model_interpolation5" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c100_wmi5_path2" 1 > log/dfl_mwi/c100_wmi5_path2.log 2>&1 &
#
#
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 10 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c10_local1_path2" 1 > log/dfl_weight/c10_local1_path2.log 2>&1 &
#nohup bash run.sh 0.01 "BaseConvNet" 1 64 100 "cuda:3" "local" "loss" "affinity_topK" "non-iid_pathological2" 100 100 999 "c100_local1_path2" 1 > log/dfl_weight/c100_local1_path2.log 2>&1 &

