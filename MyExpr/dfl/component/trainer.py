import torch.nn as nn
import torch
import wandb

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

        # todo pathological和latent的逻辑需要整理
        # pathological
        # 每个client包含的类别
        self.client_class_dic = None
        # 包含特定类别的client集合
        self.class_client_dic = None
        # 记录相互重叠的类集合
        self.overlap_client = {}

        # latent
        self.dist_client_dict = None

        # 如果是中心化联邦学习算法，需要一个中心服务器
        self.central_server = None

        # 训练策略
        self.train = None
        self.strategy = None
        self.use(args.trainer_strategy)
        # non_iid测试方法
        if args.data_distribution == "non-iid_pathological":
            self.non_iid_test = self.non_iid_test_pathological
        elif args.data_distribution == "non-iid_latent":
            self.non_iid_test = self.non_iid_test_latent

        # decrapted
        self.distributions = None

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
        elif strategy == "oracle":
            self.train = self.oracle
        elif strategy == "fedavg":
            self.central_server = Server(self.args)
            self.train = self.fedavg

    ###########
    # 基础方法 #
    ##########

    # 广播
    def broadcast(self):
        # print("-----开始广播-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.broadcast()
            # print("client {} 广播完毕".format(c_id))
        # print(self.client_dic[0].broadcaster.p, self.client_dic[0].broadcaster.w)
        # print("-----广播结束-----")

    # 选择topk
    def select_topK(self):
        # print("-----开始选择topK-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.select_topK_epoch_wise()
        # print("-----选择topK结束-----")

    # 本地训练
    def local(self):
        rounds = self.recorder.rounds
        # print("-----开始本地训练-----")
        total_loss, total_correct = 0.0, 0.0
        total_batch_num = 0
        for c_id in self.client_dic:
            # print("{}：开始训练".format(c_id))
            loss, correct = self.client_dic[c_id].local_train()
            total_loss += loss
            total_correct += correct
            total_batch_num += len(self.client_dic[c_id].train_loader)
        # print("-----本地训练结束-----")

        avg_local_train_loss, local_train_acc = total_loss / total_batch_num, total_correct / (
                    total_batch_num * self.args.batch_size)

        print("avg_local_train_loss:{}, local_train_acc:{}".
              format(avg_local_train_loss, local_train_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_local_train_loss": avg_local_train_loss, "local_train_acc": local_train_acc})

    # 互学习更新
    def mutual_update(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_batch_num = 0
        # print("-----开始深度互学习-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            loss, correct = client.deep_mutual_update_epoch_wise()
            total_loss += loss
            total_correct += correct
            total_batch_num += len(self.client_dic[c_id].train_loader)
        # 无论几轮local_train都是一轮mutual train
        # print("-----深度互学习结束-----")
        avg_mutual_train_loss = total_loss / total_batch_num
        mutual_train_acc = total_correct / (total_batch_num * self.args.batch_size)

        print("avg_mutual_train_loss:{}, mutual_train_acc:{}".
              format(avg_mutual_train_loss, mutual_train_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_mutual_train_loss": avg_mutual_train_loss,
                                         "mutual_train_acc": mutual_train_acc})

    # 缓存本轮接收到的模型（直接覆盖旧数据）
    def cache_received(self):
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.last_received_model_dict = client.received_model_dict
            client.last_received_topology_weight_dict = client.received_topology_weight_dict

    # 清空client的无用缓存(本轮topK筛选后的received_model_list)
    def clear_cache(self):
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.received_model_dict = {}
            client.received_topology_weight_dict = {}

    ###########
    # 方法组合 #
    ##########

    # 本地训练 + 互学习
    def local_and_mutual_epoch(self):
        # 本地训练
        rounds = self.recorder.rounds
        if rounds >= self.args.local_train_stop_point:
            print("本地训练已于第{}个communication_round停止".format(self.args.local_train_stop_point))
        else:
            self.local()

        self.broadcast()

        # 在这里缓存received_model，避免后续被topK删减
        self.cache_received()

        self.select_topK()

        self.mutual_update()

        # 因为已经缓存过了，这里只清空本轮记录
        self.clear_cache()

    # 仅本地训练
    def mutual(self):
        # 广播
        rounds = self.recorder.rounds

        self.broadcast()

        # 在这里缓存received，避免后续被topK删减
        self.cache_received()

        self.select_topK()

        self.mutual_update()

        # 因为已经缓存过了，这里只清空本轮记录
        self.clear_cache()

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

        # 因为已经缓存过了，这里只清空本轮记录
        self.clear_cache()

    def oracle(self):
        rounds = self.recorder.rounds

        # print("-----开始本地训练-----")
        self.local()
        # print("-----本地训练结束-----")

        total_loss, total_correct = 0.0, 0.0
        total_batch_num = 0
        # 有重叠类别的client的model进行互学习
        for c_id in self.client_dic:
            # 初始化
            if c_id not in self.overlap_client:
                self.overlap_client[c_id] = []
                for label in self.client_class_dic[c_id]:
                    for client in self.class_client_dic[label]:
                        if client != c_id and client not in self.overlap_client[c_id]:
                            self.overlap_client[c_id].append(client)
            total_batch_num += len(self.client_dic[c_id].train_loader)

            # print("client {} overlap client list:{}".format(c_id, self.overlap_client[c_id]))
            for id in self.overlap_client[c_id]:
                # self.client_dic[c_id].received_model_dict[id] = self.client_dic[id].model
                self.broadcaster.receive_from_neighbors(id, self.client_dic[id].model, c_id, self.recorder.topology_manager.get_symmetric_neighbor_list(id)[c_id])
            # print("client[0] received {}".format(self.client_dic[0].received_topology_weight_dict))

            loss, correct = self.client_dic[c_id].deep_mutual_update_epoch_wise()
            total_loss += loss
            total_correct += correct

        self.clear_cache()

        avg_mutual_train_loss = total_loss / total_batch_num
        mutual_train_acc = total_correct / (total_batch_num * self.args.batch_size)

        print("avg_mutual_train_loss:{}, mutual_train_acc:{}".
              format(avg_mutual_train_loss, mutual_train_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds,
                      data={"avg_mutual_train_loss": avg_mutual_train_loss, "mutual_train_acc": mutual_train_acc})

    def fedavg(self):
        # local_train
        self.local()

        # fedavg
        server = self.central_server
        server.aggregate()

    # test local per epoch
    def local_test(self):
        rounds = self.recorder.rounds
        total_loss = 0.0
        total_correct = 0.0

        total = 0

        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            # print("client {} contains {} test data".format(c_id, len(client.test_loader)))
            total += len(client.test_loader)
            with torch.no_grad():
                for _, (test_X, test_Y) in enumerate(client.test_loader):
                    test_X, test_Y = test_X.to(self.args.device), test_Y.to(self.args.device)
                    loss, correct = client.test(test_X, test_Y)
                    total_loss += loss.item()
                    total_correct += correct

        avg_loss = total_loss / total
        avg_acc = total_correct / (total * self.args.batch_size)
        print("avg_local_test_loss:{}, avg_local_test_acc:{}".format(avg_loss, avg_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_local_test_loss": avg_loss, "avg_local_test_acc": avg_acc})
        return avg_loss, avg_acc
        # self.recorder.record_local_test(avg_loss, avg_acc)

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
        avg_loss = total_loss / (len(self.test_loader) * len(self.client_dic))
        avg_acc = total_correct / (len(self.test_loader) * len(self.client_dic) * self.args.batch_size)
        print("avg_overall_test_loss:{}, avg_overall_test_acc:{}".format(avg_loss, avg_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_overall_test_loss": avg_loss, "avg_overall_test_acc": avg_acc})

    def non_iid_test_pathological(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_batch_sum = 0

        with torch.no_grad():
            for label in self.class_client_dic:
                # print("label{}具有{}条数据，{}个client包含这个label".format(label, len(self.test_non_iid[label]), len(self.class_client_dic[label])))
                # print("class_client_dic[label] : {}".format(self.class_client_dic[label]))

                total_batch_sum += len(self.test_non_iid[label]) * len(self.class_client_dic[label])

                for _, (test_X, test_Y) in enumerate(self.test_non_iid[label]):
                    # print("test_Y :{}".format(test_Y.detach().numpy()))
                    test_X, test_Y = test_X.to(self.args.device), test_Y.to(self.args.device)
                    for c_id in self.class_client_dic[label]:
                        client = self.client_dic[c_id]
                        loss, correct = client.test(test_X, test_Y)
                        total_loss += loss.item()
                        total_correct += correct


        # shard代表client包含的类数量
        avg_loss = total_loss / total_batch_sum
        avg_acc = total_correct / (total_batch_sum * self.args.batch_size)
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_non-iid_test_loss": avg_loss, "avg_non-iid_test_acc": avg_acc})
        print("avg_non-iid_test_loss:{}, avg_non-iid_test_acc:{}".format(avg_loss, avg_acc))

    def non_iid_test_latent(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_batch_sum = 0

        for dist in range(self.args.num_distributions):
            client_list = self.dist_client_dict[dist]
            total_batch_sum += len(self.test_non_iid[dist]) * len(client_list)
            with torch.no_grad():
                for _, (test_X, test_Y) in enumerate(self.test_non_iid[dist]):
                    test_X, test_Y =  test_X.to(self.args.device), test_Y.to(self.args.device)
                    for c_id in client_list:
                        client = self.client_dic[c_id]
                        loss, correct = client.test(test_X, test_Y)
                        total_loss += loss.item()
                        total_correct += correct

        avg_loss = total_loss / total_batch_sum
        avg_acc = total_correct / (total_batch_sum * self.args.batch_size)
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"avg_non-iid_test_loss": avg_loss, "avg_non-iid_test_acc": avg_acc})
        print("avg_non-iid_test_loss:{}, avg_non-iid_test_acc:{}".format(avg_loss, avg_acc))



