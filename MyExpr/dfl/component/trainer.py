import torch.nn as nn


# 协同训练器：采用某种协同训练策略, 控制某一轮训练
#



class Trainer(object):

    def __init__(self, recorder):
        self.recorder = recorder
        self.client_dic = recorder.client_dic
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')
        self.train = None

    def use(self, strategy):
        if strategy == "local_and_mutual":
            self.train = self.local_and_mutual_learning_collaborate_update

    # （1）进行一轮训练后再进行协同训练
    def local_and_mutual_learning_collaborate_update(self, iteration):
        client_dic = self.client_dic
        # local train
        for c_id in client_dic.keys():
            client_dic[c_id].train(iteration)
        # send model
        for c_id in client_dic.keys():
            client_dic[c_id].broadcast()
        # select top_K
        for c_id in client_dic.keys():
            client_dic[c_id].select_topK()
        # mutual_learning
        for c_id in self.client_dic.keys():
            client_dic[c_id].deep_mutual_update(iteration)
        self.recorder.record_regret()
        self.recorder.next_iteration()

    # todo （2）直接进行协同训练
