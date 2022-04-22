# 2022.4.20

# 探究threshold的影响

# dataset:latent2, local_train_epoch=5
# 170268 447293 1625
# log/expr3/c60_local_non-iid_latent2_nd2
bash run_expr.sh "TFConvNet" "cuda:2" "local" "affinity_topK" 5 60 0.1 "non-iid_latent2" 150 0.5 2 0.1 0 "expr3"
# log/expr3/c60_weighted_model_interpolation3_non-iid_latent2_nd2_th0.1
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 5 60 0.1 "non-iid_latent2" \
150 0.5 2 0.1 0 "expr3"
# log/expr3/c60_weighted_model_interpolation3_non-iid_latent2_nd2_th0.5
bash run_expr.sh "TFConvNet" "cuda:2" "weighted_model_interpolation3" "affinity_topK" 5 60 0.1 "non-iid_latent2" \
150 0.5 2 0.5 0 "expr3"



# 预训练并保存 （采用预训练模型就不用将local_train_epoch设置得太大了）
bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.01 "non-iid_latent2" 2 10 "expr3"

nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 2 100 "expr3" \
> log/pretrain/c60_non-iid_latent2_nd2_pe100.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:2" "local" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_local_latent2" 0.5 2 > log/expr3/c60_local_latent2_150.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours05_latent2_th01" 0.5 2 > log/expr3/c60_ours05_latent2_th01_150.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours05_latent2_th02" 0.5 2 > log/expr3/c60_ours05_latent2_th02_150.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours05_latent2_th05" 0.5 2 > log/expr3/c60_ours05_latent2_th05_150.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:3" "weighted_model_interpolation3" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_ours05_latent2_th01_pretrain" 0.5 2 \
> log/expr3/c60_ours05_latent2_th01_150_pretrain.log 2>&1 &

nohup bash run.sh 0.01 "TFConvNet" 5 64 60 "cuda:3" "local" "loss" "affinity_topK" \
"non-iid_latent2" 150 100 999 "c60_local_latent2_th01_pretrain" 0.5 2 \
> log/expr3/c60_local_latent2_th01_150_pretrain.log 2>&1 &




# 当前划分，后续实验有需要就修改
# latent2 dist_num=2
#Distribution 0 train (labels, counts)
#( 0, 4048) ( 1, 4984) ( 2,   15) ( 3,   10) ( 4,    2) ( 5,    2) ( 6,   14) ( 7,   23) ( 8, 4907) ( 9, 4987)
#Distribution 0 test (labels, counts)
#( 0,  770) ( 1,  998) ( 2,    2) ( 3,    6) ( 5,    2) ( 6,    5) ( 7,    5) ( 8,  985) ( 9,  998)
#Distribution 1 train (labels, counts)
#( 0,  952) ( 1,   16) ( 2, 4985) ( 3, 4990) ( 4, 4998) ( 5, 4998) ( 6, 4986) ( 7, 4977) ( 8,   93) ( 9,   13)
#Distribution 1 test (labels, counts)
#( 0,  230) ( 1,    2) ( 2,  998) ( 3,  994) ( 4, 1000) ( 5,  998) ( 6,  995) ( 7,  995) ( 8,   15) ( 9,    2)
