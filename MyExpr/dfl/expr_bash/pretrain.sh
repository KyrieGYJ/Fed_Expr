
# ---------------------------------------------------- pretrain -----------------------------------------------------------
# ------------------------- client_num:60, pretrain_epoch:{100}, dataset:latent2 {2, 3, 4, 5}------------------------------
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 2 100 "pretrain" \
> log/pretrain/c60_non-iid_latent2_nd2_pe100.log 2>&1 & # done
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 3 100 "pretrain" \
> log/pretrain/c60_non-iid_latent2_nd3_pe100.log 2>&1 &
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 4 100 "pretrain" \
> log/pretrain/c60_non-iid_latent2_nd4_pe100.log 2>&1 &
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 60 0.1 "non-iid_latent2" 5 100 "pretrain" \
> log/pretrain/c60_non-iid_latent2_nd5_pe100.log 2>&1 &

# ------------------------- client_num:{20, 100}, pretrain_epoch:{100}, dataset:path ---------------------------------
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 20 0.1 "non-iid_pathological" 2 100 "pretrain" \
> log/pretrain/c60_non-iid_pathological_nd2_pe100.log 2>&1 &
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 100 0.1 "non-iid_pathological" 2 100 "pretrain" \
> log/pretrain/c60_non-iid_pathological_nd2_pe100.log 2>&1 &

# ------------------------- client_num:{20, 100}, pretrain_epoch:{100}, dataset:path2 --------------------------------
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 20 0.1 "non-iid_pathological2" 2 100 "pretrain" \
> log/pretrain/c60_non-iid_latent2_nd2_pe100.log 2>&1 &
nohup bash run_pretrain.sh "TFConvNet" "cuda:3" 100 0.1 "non-iid_pathological2" 2 100 "pretrain" \
> log/pretrain/c60_non-iid_latent2_nd2_pe100.log 2>&1 &
# ---------------------------------------------------- pretrain ----------------------------------------------------------
