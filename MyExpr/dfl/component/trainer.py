import torch.nn as nn


# 协同训练器：采用某种协同训练策略, 控制某一次训练
# 写错了。。。
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
        if self.args.communication_wise == "epoch":
            if strategy == "local_and_mutual":
                self.train = self.local_and_mutual_epoch
        else:
            if strategy == "local_and_mutual":
                self.train = self.local_and_mutual_learning_collaborate_update
            elif strategy == "mutual":
                self.train = self.mutual_learning_collaborate_update
            elif strategy == "local_train":
                self.train = self.local_train
            elif strategy == "model_interpolation":
                self.train = self.model_interpolation

    def set_test_loader(self, test_loader):
        self.test_loader = test_loader

    ###############################
    # 1 communication per E epoch #
    ###############################
    def local_and_mutual_epoch(self):
        # 本地训练
        print("-----开始本地训练-----")
        total_loss, total_correct = 0, 0
        for c_id in self.client_dic:
            print("{}：开始训练".format(c_id))
            loss, correct = self.client_dic[c_id].local_train()
            total_loss += loss
            total_correct += correct
        print("-----本地训练结束-----")

        total_smaple_num = len(self.client_dic) * self.train_iteration * self.batch_size
        avg_local_train_loss, local_train_acc = total_loss / total_smaple_num, total_correct / total_smaple_num

        # 广播
        print("-----开始广播-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.broadcast()
        print("-----广播结束-----")

        # 选择topk
        print("-----选择topK-----")
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            client.select_topK_epoch_wise()
        print("-----选择topK结束-----")
        # mutual_update
        total_loss, total_correct = 0, 0
        for c_id in self.client_dic:
            client = self.client_dic[c_id]
            loss, correct = client.deep_mutual_update_epoch_wise()
            total_loss += loss
            total_correct += correct
        total_smaple_num = len(self.client_dic) * self.train_iteration * self.batch_size
        avg_mutual_train_loss, mutual_train_acc = total_loss / total_smaple_num, total_correct / total_smaple_num
        return avg_local_train_loss, local_train_acc, avg_mutual_train_loss, mutual_train_acc

    #################################
    # 1 communication per iteration #
    #################################
    # todo 不能为每个client打开一个迭代器，系统资源浪费
    # 刷新所有client的dataloader迭代器
    def next_epoch(self):
        for c_id in self.client_dic.keys():
            self.client_dic[c_id].refresh_train_it()

    # 清空每个client本轮收到的模型等数据
    def clear_cache(self):
        for c_id in self.client_dic.keys():
            self.client_dic[c_id].clear_cache()

    # (1) 一次本地训练+协同训练
    def local_and_mutual_learning_collaborate_update(self):
        # 所有client的local_train
        total_local_train_loss = 0
        total_local_train_correct = 0
        # 所有client的mutual_train
        total_mutual_train_loss = 0

        # ======== 这一块有点臭，因为想按batch读还是得用迭代器，这就涉及需要保存读取出来的数据，========
        # ======== 将这个过程分散到client中比较麻烦，不如直接在trainer中一起读出集中管理 ============
        client_data_dic = {}
        # load data
        for c_id in self.client_dic.keys():
            it = self.client_dic[c_id].train_it
            _, (x, y) = next(it)
            client_data_dic[c_id] = [x, y]
        # local train
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            local_train_loss, local_train_correct = client.train(client_data_dic[c_id][0],
                                                                                client_data_dic[c_id][1])
            total_local_train_loss += local_train_loss
            total_local_train_correct += local_train_correct

        # send model
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            client.broadcast()

        # select top_K
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            client.select_topK(client_data_dic[c_id][0], client_data_dic[c_id][1])

        # mutual_learning
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            mututal_train_loss = client.deep_mutual_update(client_data_dic[c_id][0], client_data_dic[c_id][1])
            total_mutual_train_loss += mututal_train_loss

        self.clear_cache()

        return total_local_train_loss, total_local_train_correct, total_mutual_train_loss
        # self.recorder.next_iteration()

    # (2) 只进行协同训练
    def mutual_learning_collaborate_update(self):
        total_mutual_train_loss = 0
        client_dic = self.client_dic
        client_data_dic = {}
        # load data
        for c_id in client_dic.keys():
            it = client_dic[c_id].train_it
            _, (x, y) = next(it)
            client_data_dic[c_id] = [x, y]
        # send model
        for c_id in client_dic.keys():
            client_dic[c_id].broadcast()
        # select top_K
        for c_id in client_dic.keys():
            client_dic[c_id].select_topK(client_data_dic[c_id][0], client_data_dic[c_id][1])
        # mutual_learning
        for c_id in self.client_dic.keys():
            mututal_train_loss = client_dic[c_id].deep_mutual_update(client_data_dic[c_id][0],
                                                                     client_data_dic[c_id][1])
            total_mutual_train_loss += mututal_train_loss
        self.clear_cache()

        return total_mutual_train_loss

    # (3) 只进行本地训练
    def local_train(self):
        # 所有client的local_train
        total_local_train_loss = 0
        total_local_train_correct = 0
        # local train
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            it = client.train_it
            _, (x, y) = next(it)
            local_train_loss, local_train_correct = self.client_dic[c_id].train(x, y)
            total_local_train_loss += local_train_loss
            total_local_train_correct += local_train_correct
        return total_local_train_loss, total_local_train_correct

    # (4) 模型插值
    def model_interpolation(self):
        # 本地训练
        total_local_train_loss, total_local_train_correct = 0, 0
        # load data
        client_data_dic = {}
        for c_id in self.client_dic.keys():
            it = self.client_dic[c_id].train_it
            _, (x, y) = next(it)
            client_data_dic[c_id] = [x, y]
        # local train
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            local_train_loss, local_train_correct = client.train(client_data_dic[c_id][0],
                                                                 client_data_dic[c_id][1])
            total_local_train_loss += local_train_loss
            total_local_train_correct += local_train_correct
        # send model
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            client.broadcast()

        # select top_K
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            client.select_topK(client_data_dic[c_id][0], client_data_dic[c_id][1])
        # 模型插值
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            client.model_interpolation_update()

        return total_local_train_loss, total_local_train_correct

    # test local per epoch
    def local_test(self):
        total_loss = 0
        total_correct = 0
        for c_id in self.client_dic.keys():
            client = self.client_dic[c_id]
            for _, (test_X, test_Y) in enumerate(client.test_loader):
                loss, correct = client.test(test_X, test_Y)
                total_loss += loss
                total_correct += correct
        avg_loss = total_loss / (len(self.client_dic) * len(self.client_dic[0].test_loader))
        avg_acc = total_correct / (len(self.client_dic) * len(self.client_dic[0].test_loader) * self.client_dic[0].test_loader.batch_size)

        return avg_loss, avg_acc
        # self.recorder.record_local_test(avg_loss, avg_acc)

    def overall_test(self):
        total_loss = 0
        total_correct = 0
        for _, (test_X, test_Y) in enumerate(self.test_loader):
            for c_id in self.client_dic.keys():
                client = self.client_dic[c_id]
                loss, correct = client.test(test_X, test_Y)
                total_loss += loss
                total_correct += correct
        avg_loss = total_loss / (len(self.test_loader) * len(self.client_dic))
        avg_acc = total_correct / (len(self.test_loader) * len(self.client_dic) * self.test_loader.batch_size)

        return avg_loss, avg_acc
        # self.recorder.record_overall_test(avg_loss, avg_acc)
