# 在pathological2划分方法下测试

# 154139 192621 444189
# log/expr4/c60_local_non-iid_pathological2_nd2
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 5 60 0.1 "non-iid_pathological2" \
150 0.5 5 0.1 0 "expr4"
# log/expr4/c60_weighted_model_interpolation3_non-iid_pathological2_nd5_th0.1
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 5 60 0.1 "non-iid_pathological2" \
150 0.5 5 0.1 0 "expr4"
# log/expr4/c60_weighted_model_interpolation3_non-iid_pathological2_nd5_th0.5
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 5 60 0.1 "non-iid_pathological2" \
150 0.5 5 0.5 0 "expr4"


nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:2" "local" "loss" "affinity_topK" \
"non-iid_pathological2" 150 100 999 "c60_local_path2" 0.5 5 > log/expr4/c60_local_path2.log 2>&1 &


nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological2" 150 100 999 "c60_ours05_path2_th01" 0.5 5 > log/expr4/c60_ours05_path2_th01.log 2>&1 &

# todo 在代码里改的，要加这个参数
nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_pathological2" 150 100 999 "c60_ours05_path2_th05" 0.5 5 > log/expr4/c60_ours05_path2_th05.log 2>&1 &

# pretrain