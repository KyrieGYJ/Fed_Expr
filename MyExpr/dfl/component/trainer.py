import torch.nn as nn
import torch
import wandb


# 协同训练器：采用某种协同训练策略, 控制某一次训练
#

class Trainer(object):

    def __init__(self, args):
        self.recorder = None
        self.client_dic = None
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')
        self.train = None
        self.test_loader = None
        # local_train的内部轮次，用于计算avg_loss
        self.train_iteration = None
        self.batch_size = None
        self.args = args

    def register_recorder(self, recorder):
        self.recorder = recorder
        self.client_dic = recorder.client_dic

    def use(self, strategy):
        description = "Trainer use strategy:{:s}"
        print(description.format(strategy))
        if strategy == "local_and_mutual":
            self.train = self.local_and_mutual_epoch
        elif strategy == "local":
            self.train = self.local
        elif strategy == "mutual":
            self.train = self.mutual
        elif strategy == "model_interpolation":
            self.train = self.model_interpolation

    def set_test_loader(self, test_loader):
        self.test_loader = test_loader

    ###############################
    # 1 communication per E epoch #
    ###############################
    def local_and_mutual_epoch(self):
        # 本地训练
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        if rounds >= self.args.local_train_stop_point:
            print("本地训练已于第{}个communication_round停止".format(self.args.local_train_stop_point))
        else:
            print("-----开始本地训练-----")
            for c_id in self.client_dic:
                print("{}：开始训练".format(c_id))
                loss, correct = self.client_dic[c_id].local_train()
                total_loss += loss
                total_correct += correct
            print("-----本地训练结束-----")
        total_batch_num = len(self.client_dic) * self.train_iteration * self.args.epochs
        avg_local_train_loss, local_train_acc = total_loss / total_batch_num, total_correct / (total_batch_num * self.batch_size)

        # 广播
        print("-----开始广播-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.broadcast()
            print("client {} 广播完毕".format(c_id))
        # print(self.client_dic[0].broadcaster.p, self.client_dic[0].broadcaster.w)
        print("-----广播结束-----")

        # 查看情况
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            print("client {}: received {} neighbor models last round".format(c_id, len(client.received_model_dict)))

        # 选择topk
        print("-----开始选择topK-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.select_topK_epoch_wise()
        print("-----选择topK结束-----")

        # mutual_update
        print("-----开始深度互学习-----")
        total_loss, total_correct = 0.0, 0.0
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            loss, correct = client.deep_mutual_update_epoch_wise()
            total_loss += loss
            total_correct += correct
        print("-----深度互学习结束-----")

        # 滑动字典，更迭上轮的接受模型
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.last_received_model_dict, client.received_model_dict = client.received_model_dict, {}
            client.last_received_topology_weight_dict, client.received_topology_weight_dict = client.received_topology_weight_dict, {}

        # 无论几轮local_train都是一轮mutual train
        # total_batch_num = len(self.client_dic) * self.train_iteration * self.args.epochs
        total_batch_num = len(self.client_dic) * self.train_iteration
        avg_mutual_train_loss = total_loss / total_batch_num
        mutual_train_acc = total_correct / (total_batch_num * self.batch_size)

        print("avg_local_train_loss:{}, local_train_acc:{}, avg_mutual_train_loss:{}, mutual_train_acc:{}".
              format(avg_local_train_loss, local_train_acc, avg_mutual_train_loss, mutual_train_acc))

        print("-----上传至wandb-----")
        wandb.log(step=rounds, data={"avg_local_train_loss": avg_local_train_loss, "local_train_acc": local_train_acc,
                                    "avg_mutual_train_loss": avg_mutual_train_loss, "mutual_train_acc": mutual_train_acc})

    def local(self):
        # 本地训练
        rounds = self.recorder.rounds
        print("-----开始本地训练-----")
        total_loss, total_correct = 0.0, 0.0
        for c_id in self.client_dic:
            print("{}：开始训练".format(c_id))
            loss, correct = self.client_dic[c_id].local_train()
            total_loss += loss
            total_correct += correct
        print("-----本地训练结束-----")

        total_batch_num = len(self.client_dic) * self.train_iteration * self.args.epochs
        avg_local_train_loss, local_train_acc = total_loss / total_batch_num, total_correct / (
                    total_batch_num * self.batch_size)

        print("avg_local_train_loss:{}, local_train_acc:{}".
              format(avg_local_train_loss, local_train_acc))

        print("-----上传至wandb-----")
        wandb.log(step=rounds, data={"avg_local_train_loss": avg_local_train_loss, "local_train_acc": local_train_acc})

    def mutual(self):
        # 广播
        rounds = self.recorder.rounds
        print("-----开始广播-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.broadcast()
            print("client {} 广播完毕".format(c_id))
        # print(self.client_dic[0].broadcaster.p, self.client_dic[0].broadcaster.w)
        print("-----广播结束-----")

        # 查看情况
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            print("client {}: received {} neighbor models last round".format(c_id, len(client.received_model_dict)))

        # 选择topk
        print("-----开始选择topK-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.select_topK_epoch_wise()
        print("-----选择topK结束-----")
        # mutual_update
        print("-----开始深度互学习-----")
        total_loss, total_correct = 0.0, 0.0
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            loss, correct = client.deep_mutual_update_epoch_wise()
            total_loss += loss
            total_correct += correct
        print("-----深度互学习结束-----")

        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.last_received_model_dict, client.received_model_dict = client.received_model_dict, {}
            client.last_received_topology_weight_dict, client.received_topology_weight_dict = client.received_topology_weight_dict, {}

        # 无论几轮local_train都是一轮mutual train
        # total_batch_num = len(self.client_dic) * self.train_iteration * self.args.epochs
        total_batch_num = len(self.client_dic) * self.train_iteration
        avg_mutual_train_loss = total_loss / total_batch_num
        mutual_train_acc = total_correct / (total_batch_num * self.batch_size)

        print("avg_mutual_train_loss:{}, mutual_train_acc:{}".
              format(avg_mutual_train_loss, mutual_train_acc))

        print("-----上传至wandb-----")
        wandb.log(step=rounds, data={"avg_mutual_train_loss": avg_mutual_train_loss, "mutual_train_acc": mutual_train_acc})

    def model_interpolation(self):
        # 本地训练
        rounds = self.recorder.rounds
        print("-----开始本地训练-----")
        total_loss, total_correct = 0.0, 0.0
        for c_id in self.client_dic:
            print("{}：开始训练".format(c_id))
            loss, correct = self.client_dic[c_id].local_train()
            total_loss += loss
            total_correct += correct
        print("-----本地训练结束-----")
        total_batch_num = len(self.client_dic) * self.train_iteration * self.args.epochs
        avg_local_train_loss, local_train_acc = total_loss / total_batch_num, total_correct / (
                    total_batch_num * self.batch_size)
        # 广播
        print("-----开始广播-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.broadcast()
            print("client {} 广播完毕".format(c_id))
        # print(self.client_dic[0].broadcaster.p, self.client_dic[0].broadcaster.w)
        print("-----广播结束-----")

        # 查看情况
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            print("client {}: received {} neighbor models last round".format(c_id, len(client.received_model_dict)))

        # 选择topk
        print("-----开始选择topK-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.select_topK_epoch_wise()
        print("-----选择topK结束-----")

        # 模型插值
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            client.model_interpolation_update()

        # 滑动字典，更迭上轮的接受模型
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.last_received_model_dict, client.received_model_dict = client.received_model_dict, {}
            client.last_received_topology_weight_dict, client.received_topology_weight_dict = client.received_topology_weight_dict, {}

        # 无论几轮local_train都是一轮mutual train
        # total_batch_num = len(self.client_dic) * self.train_iteration * self.args.epochs
        total_batch_num = len(self.client_dic) * self.train_iteration
        avg_mutual_train_loss = total_loss / total_batch_num
        mutual_train_acc = total_correct / (total_batch_num * self.batch_size)

        print("avg_local_train_loss:{}, local_train_acc:{}".format(avg_local_train_loss, local_train_acc))

        print("-----上传至wandb-----")
        wandb.log(step=rounds, data={"avg_local_train_loss": avg_local_train_loss, "local_train_acc": local_train_acc})

    # test local per epoch
    def local_test(self):
        rounds = self.recorder.rounds
        total_loss = 0.0
        total_correct = 0.0

        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            with torch.no_grad():
                for _, (test_X, test_Y) in enumerate(client.test_loader):
                    test_X, test_Y = test_X.to(self.args.device), test_Y.to(self.args.device)
                    loss, correct = client.test(test_X, test_Y)
                    total_loss += loss.item()
                    total_correct += correct

        avg_loss = total_loss / (len(self.client_dic) * len(self.client_dic[0].test_loader))
        avg_acc = total_correct / (len(self.client_dic) * len(self.client_dic[0].test_loader) * self.client_dic[0].test_loader.batch_size)
        print("avg_local_test_loss:{}, avg_local_test_acc:{}".format(avg_loss, avg_acc))

        print("-----上传至wandb-----")
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
        avg_acc = total_correct / (len(self.test_loader) * len(self.client_dic) * self.test_loader.batch_size)
        print("avg_overall_test_loss:{}, avg_overall_test_acc:{}".format(avg_loss, avg_acc))

        print("-----上传至wandb-----")
        wandb.log(step=rounds, data={"avg_overall_test_loss": avg_loss, "avg_overall_test_acc": avg_acc})
