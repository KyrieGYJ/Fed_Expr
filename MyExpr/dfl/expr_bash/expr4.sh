# 在pathological2划分方法下测试

# log/expr4/c60_local_non-iid_pathological2_nd2
bash run_expr.sh "TFConvNet" "cuda:3" "local" "affinity_topK" 5 60 0.1 "non-iid_pathological2" \
150 0.5 5 0.1 0 "expr4"
# log/expr4/c60_weighted_model_interpolation3_non-iid_pathological2_nd5_th0.1
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 5 60 0.1 "non-iid_pathological2" \
150 0.5 5 0.1 0 "expr4"
# log/expr4/c60_weighted_model_interpolation3_non-iid_pathological2_nd5_th0.5
bash run_expr.sh "TFConvNet" "cuda:3" "weighted_model_interpolation3" "affinity_topK" 5 60 0.1 "non-iid_pathological2" \
150 0.5 5 0.5 0 "expr4"


