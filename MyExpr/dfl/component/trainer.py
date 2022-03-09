import torch.nn as nn


# 协同训练器：采用某种协同训练策略, 控制某一次训练
#

class Trainer(object):

    def __init__(self):
        self.recorder = None
        self.client_dic = None
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')
        self.train = None

    def register_recorder(self, recorder):
        self.recorder = recorder
        self.client_dic = recorder.client_dic

    def use(self, strategy):
        description = "Trainer use strategy:{:s}"
        print(description.format(strategy))
        if strategy == "local_and_mutual":
            self.train = self.local_and_mutual_learning_collaborate_update

    def next_epoch(self):
        for c_id in self.client_dic.keys():
            self.client_dic[c_id].refresh_it()

    # （1）一次训练+协同训练
    def local_and_mutual_learning_collaborate_update(self):
        client_dic = self.client_dic
        # ======== 这一块有点臭，因为想按batch读还是得用迭代器，这就涉及需要保存读取出来的数据，========
        # ======== 将这个过程分散到client中比较麻烦，不如直接在trainer中一起读出集中管理 ============
        client_data_dic = {}
        # load data
        for c_id in client_dic.keys():
            it = client_dic[c_id].it
            x, y = next(it)
            client_data_dic[c_id] = [x, y]
        # ================================================================================
        # local train
        for c_id in client_dic.keys():
            client_dic[c_id].train(client_data_dic[c_id][0], client_data_dic[c_id][1])
        # send model
        for c_id in client_dic.keys():
            client_dic[c_id].broadcast()
        # select top_K
        for c_id in client_dic.keys():
            client_dic[c_id].select_topK()
        # mutual_learning
        for c_id in self.client_dic.keys():
            client_dic[c_id].deep_mutual_update(client_data_dic[c_id][0], client_data_dic[c_id][1])
        self.recorder.record_regret()
        self.recorder.next_iteration()

    # todo （2）直接进行协同训练
