import torch
import torch.nn as nn
import logging

class Client(object):
    def __init__(self, model, client_id, args, train_loader, test_loader):
        self.model = model
        self.client_id = client_id
        self.data = None
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion_CE = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

        self.args = args
        # self.device = args.device
        self.it = train_loader.__iter__()

        # 组件
        # top_K选择器，搭载多个方法
        self.topK_selector = None
        # 记录器，记录每一轮每个client的训练，测试数据
        self.recorder = None
        # 广播器，接收邻居的模型，发送自己模型
        self.broadcaster = None

        # 中间值存储
        # 当前回合邻居发送过来的模型
        self.neighbors_weight_dict = {}
        # 当前回合邻居发送过来的，在邻居的topology中，邻居->本地的权重。
        self.neighbors_topology_weight_dict = {}
        # 当前回合的topK
        self.topK = []
        # 当前回合邻居模型在本地训练数据上的output(如果加入了topK策略，output就都已经计算好了)
        self.neighbors_output_dict = {}

    def register(self, topK_selector, recorder, broadcaster):
        self.topK_selector = topK_selector
        self.recorder = recorder
        self.broadcaster = broadcaster

    def refresh_it(self):
        del self.it
        self.it = self.train_loader.__iter__()

    # 训练本地模型
    def train(self, train_X, train_Y):
        self.optimizer.zero_grad()
        outputs = self.model(train_X)
        loss = self.criterion_CE(outputs, train_Y)
        loss.backward()
        self.optimizer.step()
        # 记录本轮local_train_loss
        self.recorder.record_local_train_loss(self.client_id, loss.detach().numpy())

    # 采用DML进行协同更新
    def deep_mutual_update(self, train_X, train_Y):
        self.optimizer.zero_grad()

        # 计算本地模型loss
        local_outputs = self.model(train_X)
        local_loss = self.criterion_CE(local_outputs, train_Y)

        # 计算接收模型loss以及散度
        KLD_loss = 0
        for neighbor_id in self.neighbors_weight_dict.keys():
            neighbor_model = self.neighbors_weight_dict[neighbor_id]
            if self.neighbors_output_dict[neighbor_id] is None:
                neighbor_outputs = neighbor_model(train_X)
            else:
                neighbor_outputs = self.neighbors_output_dict[neighbor_id]
            KLD_loss += self.criterion_KLD(torch.log_softmax(neighbor_outputs, dim=1), torch.softmax(local_outputs, dim=1)).item()
        local_loss += KLD_loss / len(self.neighbors_weight_dict)

        # 反向传播 + 更新梯度
        local_loss.backward()
        self.optimizer.step()

        # 记录本轮的mutual_train_loss
        self.recorder.record_mutual_train_loss(self.client_id, local_loss.detach().numpy())

    # topK selector
    def select_topK(self):
        topK = self.topK_selector.select()
        selected_weight_dict = {}
        for c_id, loss, output in topK:
            selected_weight_dict[c_id] = self.neighbors_weight_dict[c_id]
            self.neighbors_output_dict[c_id] = output
        self.neighbors_weight_dict = selected_weight_dict

    # 广播模块
    def broadcast(self):
        self.broadcaster.send(self.client_id, self.model)

    def response(self, sender_id):
        logging.info("client[{:s}]:收到来自client[{:s}]的广播".format(self.client_id, sender_id))




