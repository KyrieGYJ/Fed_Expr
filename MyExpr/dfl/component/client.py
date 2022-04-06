import torch
import torch.nn as nn
import numpy as np
from opacus import PrivacyEngine
import logging

class Client(object):
    def __init__(self, model, client_id, args, train_loader, validation_loader, test_loader):
        self.args = args
        self.client_id = client_id
        self.data = None

        model = model.to(args.device)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        self.privacy_engine = None
        if self.args.enable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm
            )
            self.privacy_engine = privacy_engine

        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.optimizer = optimizer


        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')



        # 用于迭代获取数据
        # self.train_it = enumerate(train_loader)
        self.train_it = None

        self.device = args.device

        # 组件
        # top_K选择器，搭载多个方法
        self.topK_selector = None
        # 记录器，记录每一轮每个client的训练，测试数据
        self.recorder = None
        # 广播器，接收邻居的模型，发送自己模型
        self.broadcaster = None

        # 中间值存储
        # 上回合的缓存
        self.last_received_model_dict = {}
        # 当前回合邻居发送过来的模型
        self.received_model_dict = {}

        # 上回合的缓存
        self.last_received_topology_weight_dict = {}
        # 当前回合邻居发送过来的，在邻居的topology中，邻居->本地的权重。
        self.received_topology_weight_dict = {}
        # 当前回合的topK
        self.topK = []
        # 当前回合邻居模型在本地训练数据上的output(如果加入了topK策略，output就都已经计算好了)
        self.neighbors_output_dict = {}

        # 上回合的缓存
        self.last_received_w_dict = {}
        # 保存其他client发送过来的weight
        self.received_w_dict = {}
        self.p = []
        self.w = []
        self.affinity_matrix = []

    def register(self, topK_selector, recorder, broadcaster):
        self.topK_selector = topK_selector
        self.recorder = recorder
        self.broadcaster = broadcaster

    def initialize(self):
        # todo 只有affinity_cluster要用到
        client_dic = self.recorder.client_dic
        other_weight = 0.
        client_initial_self_weight = 0.1
        # self.p = [[torch.tensor(1. * other_weight) for _ in client_dic] for _ in client_dic]
        # self.w = [[torch.tensor(1. * other_weight) for _ in client_dic] for _ in client_dic]
        self.p = np.ones([self.args.client_num_in_total, self.args.client_num_in_total], dtype=np.float64) * other_weight
        self.w = np.ones([self.args.client_num_in_total], dtype=np.float64) * other_weight
        for c_id in client_dic:
            topology = self.recorder.topology_manager.get_symmetric_neighbor_list(c_id)
            for neighbor_id in client_dic:
                # 不相邻的client不存在权重
                if topology[neighbor_id] == 0:
                    self.p[c_id][neighbor_id] = 0.
                    self.w[neighbor_id] = 0.
                elif neighbor_id == c_id:
                    self.p[c_id][neighbor_id] = 1. * client_initial_self_weight
                    self.w[neighbor_id] = 1. * client_initial_self_weight

    def local_train(self):
        self.model.train()
        total_loss, total_correct = 0.0, 0.0
        for epoch in range(self.args.epochs):
            iteration = 0
            for idx, (train_X, train_Y) in enumerate(self.train_loader):
                train_loss, correct = self.train(train_X, train_Y)
                total_loss += train_loss
                total_correct += correct
                iteration += 1
            # print("client {}: local train takes {} iteration".format(self.client_id, iteration))
        if self.args.enable_dp:
            epsilon, best_alpha = self.privacy_engine.accountant.get_privacy_spent(
                delta=self.args.delta
            )
            print(
                f"Client: {self.client_id} \t"
                f"Loss: {np.mean(total_loss):.6f} "
                f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
            )
            return total_loss, total_correct, epsilon, best_alpha
        return total_loss, total_correct

    # 训练本地模型
    def train(self, train_X, train_Y):
        self.optimizer.zero_grad()
        train_X, train_Y = train_X.to(self.device), train_Y.to(self.device)
        outputs = self.model(train_X)
        loss = self.criterion_CE(outputs, train_Y)

        pred = outputs.argmax(dim=1)
        correct = pred.eq(train_Y.view_as(pred)).sum()

        loss.backward()
        self.optimizer.step()

        # 记录本轮local_train_loss
        if "cuda" in self.device:
            loss = loss.cpu()
        # self.recorder.record_local_train_loss(self.client_id, loss.detach().numpy())
        # self.recorder.record_local_train_correct(self.client_id, correct)


        return loss.detach().numpy(), correct

    def deep_mutual_update_epoch_wise(self):
        self.model.train()
        total_loss, total_correct = 0.0, 0.0
        total_KLD_loss = 0.0
        total_local_loss = 0.0
        iteration = 0
        # print(f"client[{self.client_id}]共{len(self.received_model_dict)}个client互学习参与")
        for idx, (train_X, train_Y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            train_X, train_Y = train_X.to(self.device), train_Y.to(self.device)
            local_outputs = self.model(train_X)
            local_loss = self.criterion_CE(local_outputs, train_Y)

            if "cuda" in self.args.device:
                local_loss = local_loss.cpu()

            total_local_loss += local_loss.item()

            pred = local_outputs.argmax(dim=1)
            correct = pred.eq(train_Y.view_as(pred)).sum()

            KLD_loss = 0
            for neighbor_id in self.received_model_dict.keys():
                neighbor_model = self.received_model_dict[neighbor_id]
                neighbor_model.eval()
                neighbor_outputs = neighbor_model(train_X)
                kld_loss = self.criterion_KLD(self.LogSoftmax(local_outputs),
                                               self.Softmax(neighbor_outputs.detach()))
                if "cuda" in self.args.device:
                    kld_loss = kld_loss.cpu()

                KLD_loss += kld_loss.item()

            if len(self.received_model_dict) > 0:
                KLD_loss = KLD_loss / len(self.received_model_dict)
                local_loss += KLD_loss

            local_loss.backward()
            self.optimizer.step()

            if "cuda" in self.device:
                local_loss = local_loss.cpu()
            total_loss += local_loss.item()
            total_correct += correct
            iteration += 1
            total_KLD_loss += KLD_loss

        if iteration > 0:
            total_KLD_loss /= iteration

        # print("client {} 的平均互学习KL散度为:{}, local_loss为:{}".format(self.client_id, total_KLD_loss, total_local_loss))
        return total_loss, total_correct


    # 采用dsgd模型插值进行协同更新
    def model_interpolation_update(self):
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(self.client_id)

        for x_paras in self.model.parameters():
            x_paras.data.mul_(topology[self.client_id])

        for client_id in self.received_model_dict.keys():
            neighbor_model = self.received_model_dict[client_id]
            topo_weight = self.received_topology_weight_dict[client_id]
            for x_paras, x_neighbor in zip(list(self.model.parameters()), list(neighbor_model.parameters())):
                temp = x_neighbor.data.mul(topo_weight)
                x_paras.data.add_(temp)

    def select_topK_epoch_wise(self):
        topK = self.topK_selector.select(self)
        if topK is None:
            return
        selected_weight_dict = {}
        for c_id, loss in topK:
            selected_weight_dict[c_id] = self.received_model_dict[c_id]
        self.received_model_dict = selected_weight_dict
        # print("client {} 本轮topK选择了{}个模型".format(self.client_id, len(self.received_model_dict)))

    # 广播模块
    def broadcast(self):
        self.broadcaster.send(self.client_id, self.model)

    def response(self, sender_id):
        # logging.info("client[{:d}]:收到来自client[{:d}]的广播".format(self.client_id, sender_id))
        pass

    def test(self, test_X, test_Y):
        # with torch.no_grad():
        #     test_X, test_Y = test_X.to(self.device), test_Y.to(self.device)
        #     outputs = self.model(test_X)
        #     loss = self.criterion_CE(outputs, test_Y)
        #
        #     pred = outputs.argmax(dim=1)
        #     correct = pred.eq(test_Y.view_as(pred)).sum()
        #     if "cuda" in self.device:
        #         loss = loss.cpu()
        #     return loss.detach().numpy(), correct
        self.model.eval()
        test_X, test_Y = test_X.to(self.device), test_Y.to(self.device)
        outputs = self.model(test_X)
        loss = self.criterion_CE(outputs, test_Y)

        pred = outputs.argmax(dim=1)
        correct = pred.eq(test_Y.view_as(pred)).sum()
        if "cuda" in self.device:
            loss = loss.cpu()
        return loss.detach().numpy(), correct



