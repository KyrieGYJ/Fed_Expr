import sys

import torch.nn as nn
import torch
import wandb
import numpy as np
from tqdm import tqdm

from MyExpr.cfl.server import Server

# 协同训练器：采用某种协同训练策略, 控制某一个communication_round训练
class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.recorder = None
        self.client_dic = None
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

        self.data = None
        self.test_loader = None
        self.broadcaster = None
        # 逐个label的测试数据
        self.test_non_iid = None
        self.non_iid_test_set = None

        # pathological
        # 每个client包含的类别
        self.client_class_dic = None
        # 包含特定类别的client集合
        self.class_client_dic = None
        # 记录相互重叠的类集合
        self.overlap_client = {}

        self.best_accuracy = 0.

        # latent
        self.dist_client_dict = None
        self.client_dist_dict = None

        # 如果是中心化联邦学习算法，需要一个中心服务器
        self.central_server = None

        # 训练策略
        self.train = None
        self.strategy = None
        self.use(args.trainer_strategy)
        # non_iid测试方法
        if args.data_distribution in ["non-iid_pathological", "non-iid_pathological2"]:
            self.non_iid_test = self.non_iid_test_pathological
        elif args.data_distribution in ["non-iid_latent", "non-iid_latent2"]:
            self.non_iid_test = self.non_iid_test_latent
        else:
            self.non_iid_test = self.non_iid_test_pass

        self.distributions = None

        # 权重更新策略（沿用，缺省）
        self.p_update_strategy = "reuse"

    def register_recorder(self, recorder):
        self.recorder = recorder
        self.client_dic = recorder.client_dic
        # todo 把这个初始化要改掉
        if self.strategy == "fedavg":
            self.central_server.client_dic = self.client_dic

    def use(self, strategy):
        description = "Trainer use strategy:{:s}"
        print(description.format(strategy))
        self.strategy = strategy
        if strategy == "local_and_mutual":
            self.train = self.local_and_mutual_epoch
        elif strategy == "local":
            self.train = self.local
        elif strategy == "mutual":
            self.train = self.mutual
        elif strategy == "model_interpolation":
            self.train = self.model_interpolation
        elif "weighted_model_interpolation" in strategy:
            self.train = self.weighted_model_interpolation
        elif strategy == "oracle_class":
            self.train = self.oracle_class
        elif strategy == "oracle_distribution":
            self.train = self.oracle_distribution
        elif strategy == "fedavg":
            self.central_server = Server(self.args)
            self.train = self.fedavg

    ###################
    # 基础方法(批量操作) #
    ###################

    # 广播
    def broadcast(self):
        for c_id in tqdm(self.client_dic, desc="broadcast"):
            self.client_dic[c_id].broadcast()

    def update_weight(self):
        for sender_id in tqdm(self.client_dic, desc="update weight"):
            self.client_dic[sender_id].cache_keeper.update_weight()

    def update_broadcast_weight(self):
        for sender_id in tqdm(self.client_dic, desc="update broadcast weight"):
            if self.recorder.rounds > 0:
                # broadcast前更新了本地模型，更新local_eval
                self.client_dic[sender_id].cache_keeper.update_local_eval()
                self.client_dic[sender_id].cache_keeper.update_broadcast_weight()
            else:
                # 第一轮还没收到模型，raw_eval_loss为空，无法更新broadcast_weight
                self.client_dic[sender_id].cache_keeper.update_local_eval()
            self.client_dic[sender_id].cache_keeper.update_p()

    def update_update_weight(self):
        for c_id in tqdm(self.client_dic, desc="update update weight"):
            self.client_dic[c_id].cache_keeper.update_raw_eval_list() # 接受到了新模型，更新eval
            self.client_dic[c_id].cache_keeper.update_update_weight()

    # 选择topk
    def select_topK(self):
        # print("-----开始选择topK-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.select_topK()
        # print("-----选择topK结束-----")

    # 本地训练
    def local(self, turn_on_wandb=True):
        rounds = self.recorder.rounds
        # print("-----开始本地训练-----")
        total_loss, total_correct = 0.0, 0.0
        total_epsilon, total_alpha = 0.0, 0.0
        total_num = 0
        for c_id in tqdm(self.client_dic, desc="local train"):
            if self.args.enable_dp:
                loss, correct, epsilon, alpha = self.client_dic[c_id].local_train()
                total_epsilon += epsilon
                total_alpha += alpha
            else:
                loss, correct = self.client_dic[c_id].local_train()
            total_loss += loss
            total_correct += correct
            total_num += len(self.client_dic[c_id].train_set)
        # print("-----本地训练结束-----")

        # print(total_num)
        local_train_acc = total_correct / total_num
        avg_local_train_epsilon, avg_local_train_alpha = total_epsilon / len(self.client_dic), total_alpha / len(self.client_dic)

        if self.args.enable_dp:
            print(f"avg_local_train_epsilon:{avg_local_train_epsilon}, avg_local_train_alpha:{avg_local_train_alpha}")

        print("local_train_loss:{}, local_train_acc:{}".
              format(total_loss, local_train_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb and turn_on_wandb:
            wandb.log(step=rounds, data={"local_train/loss": total_loss, "local_train/acc": local_train_acc})
            if self.args.enable_dp:
                wandb.log(step=rounds, data={"avg_local_train_epsilon": avg_local_train_epsilon, "avg_local_train_alpha":avg_local_train_alpha})

    # 互学习更新
    def mutual_update(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_local_loss, total_KLD_loss = 0.0, 0.0
        total_num = 0
        # print("-----开始深度互学习-----")
        for c_id in tqdm(self.client_dic, desc="mutual train"):
            client = self.client_dic[c_id]
            loss, correct,  local_loss, KLD_loss = client.deep_mutual_update()
            # print(f"trainer: client {c_id} choose neighbors {client.received_model_dict.keys()}, local_loss:{local_loss}, KLD_loss:{KLD_loss}")
            total_loss += loss
            total_correct += correct
            total_local_loss += local_loss
            total_KLD_loss += KLD_loss
            total_num += len(self.client_dic[c_id].train_set)
        # 无论几轮local_train都是一轮mutual train
        # print("-----深度互学习结束-----")
        mutual_train_loss, mutual_train_acc = total_loss, total_correct / total_num

        print(f"mutual_total_loss:{mutual_train_loss}, mutual_local_loss:{total_local_loss}, mutual_KLD_loss:{total_KLD_loss}, mutual_train_acc:{mutual_train_acc}")

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"mutual/total_loss": mutual_train_loss,
                                         "mutual/local_loss": total_local_loss,
                                         "mutual/KLD_loss": total_KLD_loss,
                                         "mutual/train_acc": mutual_train_acc})

    # 权重插值
    def weighted_interpolation_update(self):
        # todo 记得改
        for c_id in tqdm(self.client_dic, desc="weighted_interpolation"):
            if self.strategy == "weighted_model_interpolation":
                self.client_dic[c_id].weighted_model_interpolation_update()
            elif self.strategy == "weighted_model_interpolation2":
                self.client_dic[c_id].weighted_model_interpolation_update2()
            elif self.strategy == "weighted_model_interpolation3":
                self.client_dic[c_id].weighted_model_interpolation_update3()
            elif self.strategy == "weighted_model_interpolation4":
                self.client_dic[c_id].weighted_model_interpolation_update4()
            elif self.strategy == "weighted_model_interpolation5":
                self.client_dic[c_id].weighted_model_interpolation_update5()

    # 缓存本地模型，以和下一轮local_train后的模型做区分
    def cache_model(self):
        for c_id in tqdm(self.client_dic, desc="cache_last_local"):
            self.client_dic[c_id].cache_keeper.cache_last_local()

    def cache_received(self):
        for c_id in self.client_dic:
            self.client_dic[c_id].cache_keeper.update_received_memory()

    def clear_received(self):
        for c_id in self.client_dic:
            self.client_dic[c_id].received_model_dict = {}
            self.client_dic[c_id].received_topology_weight_dict = {}
            self.client_dic[c_id].received_w_dict = {}

    ###########
    # 方法组合 #
    ##########

    # 本地训练 + 互学习
    def local_and_mutual_epoch(self):
        # 本地训练
        rounds = self.recorder.rounds

        # if rounds >= self.args.local_train_stop_point:
        #     print("本地训练已于第{}个communication_round停止".format(self.args.local_train_stop_point))
        # else:
        #     self.local()

        self.local()

        self.broadcast()

        # 在这里缓存received_model，避免后续被topK删减
        self.cache_received()

        # self.select_topK()

        self.mutual_update()

        # 因为已经缓存过了，这里只清空本轮记录
        self.clear_received()

    # 仅进行互学习
    def mutual(self):
        # 广播
        rounds = self.recorder.rounds

        self.broadcast()

        # 在这里缓存received，避免后续被topK删减
        # self.cache_received()

        # self.select_topK()

        self.mutual_update()

        # 因为已经缓存过了，这里只清空本轮记录
        # self.clear_cache()

    # 模型插值
    def model_interpolation(self):
        rounds = self.recorder.rounds
        # 本地训练
        self.local()

        # 广播
        self.broadcast()

        # 在这里缓存received，避免后续被topK删减
        self.cache_received()

        # 选择topk
        self.select_topK()

        # 模型插值
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            client.model_interpolation_update()

        # clear_received，这里只清空本轮记录
        self.clear_cache()

    # 权重模型插值
    def weighted_model_interpolation(self):
        self.cache_model() # 记录local train之前的model
        self.local()
        self.update_broadcast_weight()
        # todo 需要对称性的话，要在这里与received_weight相加取平均。
        self.broadcast()
        self.cache_received()
        self.update_update_weight()
        self.weighted_interpolation_update()
        self.clear_received()

    # Pens


    # push-sum

    # gossip



    # 只跟标签重叠的client通信
    def oracle_class(self):
        rounds = self.recorder.rounds
        self.local()

        total_loss, total_correct = 0.0, 0.0
        total_num = 0

        # 向标签有重叠的client发送模型
        for c_id in self.client_dic:
            # 初始化
            if c_id not in self.overlap_client:
                self.overlap_client[c_id] = []
                for label in self.client_class_dic[c_id]:
                    for client in self.class_client_dic[label]:
                        if client != c_id and client not in self.overlap_client[c_id]:
                            self.overlap_client[c_id].append(client)
            total_num += len(self.client_dic[c_id].train_set)

            # print("client {} overlap client list:{}".format(c_id, self.overlap_client[c_id]))
            for id in self.overlap_client[c_id]:
                # self.client_dic[c_id].received_model_dict[id] = self.client_dic[id].model
                self.broadcaster.receive_from_neighbors(id, self.client_dic[id].model, c_id,
                                                        self.recorder.topology_manager.get_symmetric_neighbor_list(c_id)[
                                                            id])
            # print("client[0] received {}".format(self.client_dic[0].received_topology_weight_dict))

        for c_id in self.client_dic:
            loss, correct = self.client_dic[c_id].deep_mutual_update()
            total_loss += loss
            total_correct += correct

        self.clear_cache()

        avg_mutual_train_loss, mutual_train_acc = total_loss / total_num, total_correct / total_num

        print("avg_mutual_train_loss:{}, mutual_train_acc:{}".
              format(avg_mutual_train_loss, mutual_train_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds,
                      data={"avg_mutual_train_loss": avg_mutual_train_loss, "mutual_train_acc": mutual_train_acc})

    # 只跟同一个分布的client通信
    def oracle_distribution(self):
        rounds = self.recorder.rounds
        self.local()
        total_loss, total_correct = 0.0, 0.0
        total_num = 0

        for sender_id in self.client_dic:
            sender = self.client_dic[sender_id]
            dist = self.dist_client_dict[self.client_dist_dict[sender_id]]
            # print(f"client {sender_id} in the same dist of {dist}")
            for neighbor_id in dist:
                self.broadcaster.receive_from_neighbors(sender_id, sender.model, neighbor_id,
                                                        self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)[
                                                            neighbor_id])
            total_num += len(self.client_dic[sender_id].train_set)

        self.mutual_update()

        self.clear_cache()

    ############
    # 中心化方法 #
    ############
    def fedavg(self):
        # local_train
        self.local()

        # fedavg
        server = self.central_server
        server.aggregate()

    ###########
    # 测试方法 #
    ##########
    # test local per epoch
    def local_test(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0., 0.
        total_num = 0

        for c_id in tqdm(self.client_dic, desc="local_test"):
            client = self.client_dic[c_id]
            loss, correct = client.local_test()
            total_loss += loss
            total_correct += correct
            # print("client {} contains {} test data".format(c_id, len(client.test_loader)))
            total_num += len(client.test_set)

        avg_acc = total_correct / total_num
        print("local_test_loss:{}, avg_local_test_acc:{}".format(total_loss, avg_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"local_test/loss": total_loss, "local_test/avg_acc": avg_acc})
            if avg_acc > self.best_accuracy:
                wandb.run.summary["best_accuracy"] = avg_acc
                self.best_accuracy = avg_acc

    def non_iid_test_latent(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_num = 0

        for dist in range(self.args.num_distributions):
            client_list = self.dist_client_dict[dist]
            total_num += len(self.non_iid_test_set[dist]) * len(client_list)
            with torch.no_grad():
                for _, (test_X, test_Y) in enumerate(self.test_non_iid[dist]):
                    test_X, test_Y =  test_X.to(self.args.device), test_Y.to(self.args.device)
                    for c_id in client_list:
                        client = self.client_dic[c_id]
                        loss, correct = client.test(test_X, test_Y)
                        total_loss += loss.item()
                        total_correct += correct

        avg_loss, avg_acc = total_loss / total_num, total_correct / total_num
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_non-iid_test_loss": avg_loss, "avg_non-iid_test_acc": avg_acc})
        print("avg_non-iid_test_loss:{}, avg_non-iid_test_acc:{}".format(avg_loss, avg_acc))

    def non_iid_test_pass(self):
        pass

    # 废弃的方法
    def overall_test(self):
        rounds = self.recorder.rounds
        total_loss = 0.0
        total_correct = 0.0

        with torch.no_grad():
            for _, (test_X, test_Y) in enumerate(self.test_loader):
                test_X, test_Y = test_X.to(self.args.device), test_Y.to(self.args.device)
                for c_id in self.client_dic.keys():
                    client = self.client_dic[c_id]
                    loss, correct = client.test(test_X, test_Y)
                    total_loss += loss
                    total_correct += correct
        avg_loss = total_loss / len(self.client_dic)
        avg_acc = total_correct / (len(self.test_data) * len(self.client_dic))
        print("avg_overall_test_loss:{}, avg_overall_test_acc:{}".format(avg_loss, avg_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_overall_test_loss": avg_loss, "avg_overall_test_acc": avg_acc})

    def non_iid_test_pathological(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_num = 0

        with torch.no_grad():
            for label in self.class_client_dic:
                # print("label{}具有{}条数据，{}个client包含这个label".format(label, len(self.test_non_iid[label]), len(self.class_client_dic[label])))
                # print("class_client_dic[label] : {}".format(self.class_client_dic[label]))

                total_num += len(self.non_iid_test_set[label]) * len(self.class_client_dic[label])

                for _, (test_X, test_Y) in enumerate(self.test_non_iid[label]):
                    # print("test_Y :{}".format(test_Y.detach().numpy()))
                    test_X, test_Y = test_X.to(self.args.device), test_Y.to(self.args.device)
                    for c_id in self.class_client_dic[label]:
                        client = self.client_dic[c_id]
                        loss, correct = client.test(test_X, test_Y)
                        total_loss += loss.item()
                        total_correct += correct

        # shard代表client包含的类数量
        avg_loss, avg_acc = total_loss / total_num, total_correct / total_num
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_non-iid_test_loss": avg_loss, "avg_non-iid_test_acc": avg_acc})
        print("avg_non-iid_test_loss:{}, avg_non-iid_test_acc:{}".format(avg_loss, avg_acc))