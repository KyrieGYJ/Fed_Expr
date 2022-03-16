#!/usr/bin/env bash

nohup bash run.sh 0.01 "resnet101" 200 64  10 "cuda:3" "mutual" "non-iid(1)" 2>&1 &

nohup bash run.sh 0.01 "resnet50" 200 64  10 "cuda:1" "mutual" "non-iid(1)" 2>&1 &

nohup bash run.sh 0.01 "resnet34" 2 64  100 "cuda:3" "local_and_mutual" "loss" "non-iid" 2>&1 &

nohup bash run.sh 0.01 "resnet18" 200 64  10 "cuda:0" "mutual" "non-iid(1)" 2>&1 &