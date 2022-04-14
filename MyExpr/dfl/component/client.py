import torch
import torch.nn as nn
import numpy as np
import copy
from opacus import PrivacyEngine
import logging
import math

# from MyExpr.utils import cal_delta_loss, cal_model_dif
from .cache_keeper import keeper


class Client(object):
    def __init__(self, model, client_id, args, data, topK_selector, recorder, broadcaster):
        self.args = args
        self.client_id = client_id
        self.data = data

        train_loader = data.train_loader[client_id]
        validation_loader = data.validation_loader[client_id]
        test_loader = data.test_loader[client_id]

        # todo 可能得改，model一直占GPU不利于大模型训练
        model = model.to(args.device)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd, amsgrad=True)

        # differential privacy
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

        self.train_set = data.train_set[client_id]
        self.test_set = data.test_set[client_id]
        self.validation_set = data.validation_set[client_id]

        self.optimizer = optimizer

        self.Softmax = nn.Softmax(dim=1)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')

        self.device = args.device

        # 组件
        self.topK_selector = topK_selector
        self.recorder = recorder
        self.broadcaster = broadcaster
        self.cache_keeper = keeper(self)

        ########################################
        # cache of current communication round #
        ########################################
        self.received_model_dict = {}
        self.received_topology_weight_dict = {}
        # broadcast weight from neighbor
        self.received_w_dict = {}

        self.topK_neighbor = None

        # self broadcast weight
        self.p = []
        # self.broadcast_w = []
        # self.update_w = []
        self.affinity_matrix = []


    ################
    # an iteration #
    ################
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
        return loss.detach().numpy(), correct

    def test(self, test_X, test_Y):
        self.model.eval()
        test_X, test_Y = test_X.to(self.device), test_Y.to(self.device)
        outputs = self.model(test_X)
        loss = self.criterion_CE(outputs, test_Y)

        pred = outputs.argmax(dim=1)
        correct = pred.eq(test_Y.view_as(pred)).sum()
        if "cuda" in self.device:
            loss = loss.cpu()
        return loss.detach().numpy(), correct

    ############
    # an epoch #
    ############
    def local_train(self):
        self.model.train()
        epochs = self.args.epochs
        total_loss, total_correct = 0.0, 0.0
        for epoch in range(epochs):
            iteration = 0
            for idx, (train_X, train_Y) in enumerate(self.train_loader):
                train_loss, correct = self.train(train_X, train_Y)
                total_loss += train_loss
                total_correct += correct
                iteration += 1
            # print("client {}: local train takes {} iteration".format(self.client_id, iteration))
        total_loss /= epochs
        total_correct /= epochs
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

    def deep_mutual_update(self):
        self.model.train()
        epochs = 1
        total_loss, total_correct = 0.0, 0.0
        total_local_loss, total_KLD_loss = 0.0, 0.0

        mutual_update_candidate = self.received_model_dict if self.topK_neighbor is None else self.topK_neighbor

        print(f"self: client {self.client_id} 中有client:[{self.received_model_dict.keys()}]参与互学习")
        for epoch in range(epochs):
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
                for neighbor_id in mutual_update_candidate.keys():
                # for neighbor_id in self.last_received_w_dict.keys():
                    if self.cache_keeper.mutual_update_weight[neighbor_id] == 0.:
                        # print(f"client {self.client_id} skip neighbor {neighbor_id}")
                        continue
                    neighbor_model = mutual_update_candidate[neighbor_id]
                    neighbor_model.eval()
                    neighbor_outputs = neighbor_model(train_X)
                    kld_loss = self.criterion_KLD(self.LogSoftmax(local_outputs),
                                                   self.Softmax(neighbor_outputs.detach()))
                    if "cuda" in self.args.device:
                        kld_loss = kld_loss.cpu()

                    KLD_loss += kld_loss.item() * self.cache_keeper.mutual_update_weight[neighbor_id]

                local_loss += KLD_loss

                local_loss.backward()
                self.optimizer.step()

                if "cuda" in self.device:
                    local_loss = local_loss.cpu()
                total_loss += local_loss.item()
                total_correct += correct
                total_KLD_loss += KLD_loss
        return total_loss / epochs, total_correct / epochs, total_local_loss / epochs, total_KLD_loss / epochs

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

    # todo 改bug
    def weighted_model_interpolation_update(self):
        local_factor = 0.5
        for x_paras in self.model.parameters():
            x_paras.data.mul_(local_factor)

        flag = False
        temp_paras = None
        total_weight = 0
        for neighbor_id in self.received_model_dict.keys():
            neighbor_model = self.received_model_dict[neighbor_id]
            neighbor_weight = self.cache_keeper.mutual_update_weight[neighbor_id]
            if neighbor_weight > 0:
                flag = True
            # for x_paras, x_neighbor in zip(list(self.model.parameters()), list(neighbor_model.parameters())):
            #     temp = x_neighbor.data.mul(neighbor_weight).mul(1 - local_factor)
            #     if temp_paras is None:
            #         temp_paras = temp
            #     else:
            #         temp_paras.data.add_(temp)
            for i, neighbor_para in enumerate(list)
                # temp = x_neighbor.data.mul(neighbor_weight).mul(1 - local_factor)
                # x_paras.data.add_(temp)
            total_weight += neighbor_weight
        if temp_paras is not None:
            temp_paras.data.div_(total_weight)
            x_paras.data.add_(temp_paras)

        if not flag:
            for x_paras in self.model.parameters():
                x_paras.data.mul_(1 / local_factor)

    # todo 临时
    def weighted_model_interpolation_update2(self):
        local_weight = self.cache_keeper.mutual_update_weight2[self.client_id]
        total_weight = local_weight
        for x_paras in self.model.parameters():
            x_paras.data.mul_(local_weight)

        for neighbor_id in self.received_model_dict.keys():
            neighbor_model = self.received_model_dict[neighbor_id]
            neighbor_weight = self.cache_keeper.mutual_update_weight2[neighbor_id]
            for x_paras, x_neighbor in zip(list(self.model.parameters()), list(neighbor_model.parameters())):
                temp = x_neighbor.data.mul(neighbor_weight)
                x_paras.data.add_(temp)
            total_weight += neighbor_weight
        x_paras.data.div_(total_weight)

    def weighted_model_interpolation_update3(self):
        local_weight = self.cache_keeper.mutual_update_weight3[self.client_id]
        total_weight = local_weight
        for x_paras in self.model.parameters():
            x_paras.data.mul_(local_weight)

        for neighbor_id in self.received_model_dict.keys():
            neighbor_model = self.received_model_dict[neighbor_id]
            neighbor_weight = self.cache_keeper.mutual_update_weight3[neighbor_id]
            for x_paras, x_neighbor in zip(list(self.model.parameters()), list(neighbor_model.parameters())):
                temp = x_neighbor.data.mul(neighbor_weight)
                x_paras.data.add_(temp)
            total_weight += neighbor_weight
        x_paras.data.div_(total_weight)

    def weighted_model_interpolation_update4(self):
        local_weight = self.cache_keeper.mutual_update_weight4[self.client_id]
        total_weight = local_weight
        for x_paras in self.model.parameters():
            x_paras.data.mul_(local_weight)

        for neighbor_id in self.received_model_dict.keys():
            neighbor_model = self.received_model_dict[neighbor_id]
            neighbor_weight = self.cache_keeper.mutual_update_weight4[neighbor_id]
            for x_paras, x_neighbor in zip(list(self.model.parameters()), list(neighbor_model.parameters())):
                temp = x_neighbor.data.mul(neighbor_weight)
                x_paras.data.add_(temp)
            total_weight += neighbor_weight
        x_paras.data.div_(total_weight)


    def select_topK(self):
        topK = self.topK_selector.select(self)
        if topK is None:
            return
        # self.topK = topK
        selected_weight_dict = {}
        # todo 改这里
        for c_id, loss in topK:
            selected_weight_dict[c_id] = self.received_model_dict[c_id]
        # self.received_model_dict = selected_weight_dict
        self.topK_neighbor = selected_weight_dict

        # print("client {} 本轮topK选择了{}个模型".format(self.client_id, len(self.received_model_dict)))

    # todo 改
    # def update_broadcast_weight(self, balanced=False, model_dif_adjust=True):
    #     new_w_dict = cal_delta_loss(self, self.recorder.args)
    #
    #     new_broadcast_w_list = []
    #     new_update_w_list = []
    #
    #     epsilon = 1e-9
    #     for i in range(self.args.client_num_in_total):
    #         # relu
    #         if i in new_w_dict and new_w_dict[i] > 0:
    #             new_update_w_list.append(copy.deepcopy(new_w_dict[i]))
    #         else:
    #             new_update_w_list.append(0)
    #
    #     norm_factor = max(np.sum(new_update_w_list), epsilon)
    #     new_update_w_list = np.array(new_update_w_list) / norm_factor
    #
    #     for i in range(self.args.client_num_in_total):
    #         if i in new_w_dict:
    #             new_broadcast_w_list.append(copy.deepcopy(new_w_dict[i]))
    #         else:
    #             # print(f"{self.client_id} 缺少 {i}")
    #             new_broadcast_w_list.append(0)
    #
    #     new_broadcast_w_list = np.array(new_broadcast_w_list)
    #
    #     if model_dif_adjust:
    #         dif_list = [0 for _ in range(self.args.client_num_in_total)]
    #         dif_dict = cal_model_dif(self, self.args, norm_type="l2_root")
    #         for c_id, dif in dif_dict.items():
    #             dif_list[c_id] = dif_dict[c_id]
    #         dif_list = np.array(dif_list)
    #         dif_list[self.client_id] = np.min(dif_list)
    #         dif_list = (dif_list - np.min(dif_list)) / (np.max(dif_list) - np.min(dif_list)) # norm
    #         dif_list = (1 - dif_list)
    #
    #         new_broadcast_w_list *= dif_list
    #         new_update_w_list *= dif_list
    #
    #     if balanced:
    #         new_broadcast_w_list /= len(self.validation_set)
    #
    #     # 试下
    #     # new_broadcast_w_list = (new_broadcast_w_list - np.min(new_broadcast_w_list)) / (np.max(new_broadcast_w_list) - np.min(new_broadcast_w_list))
    #
    #     # # norm 1
    #     # normalization_factor = np.abs(np.sum(new_broadcast_w_list))
    #     #
    #     # if normalization_factor < 1e-9:
    #     #     print('Normalization factor is really small')
    #     #     normalization_factor += 1e-9
    #     # new_broadcast_w_list = np.array(new_broadcast_w_list) / normalization_factor
    #
    #     norm_factor = np.sum(new_update_w_list)
    #     norm_factor = max(norm_factor, epsilon)
    #     new_update_w_list = np.array(new_update_w_list) / norm_factor
    #
    #     print(f"client {self.client_id} new_update_w_list : {new_update_w_list}")
    #     print(f"client {self.client_id} new_broadcast_w_list : {new_broadcast_w_list}")
    #
    #     # 更新权重
    #     self.update_w = new_update_w_list
    #     self.broadcast_w = new_broadcast_w_list
    #
    # def update_p(self, self_max=True):
    #     self.p[self.client_id] += self.broadcast_w
    #     for neighbor_id in self.received_w_dict:
    #         self.p[neighbor_id] += self.received_w_dict[neighbor_id]
    #
    #     # 固定自身权重为最高
    #     if self_max:
    #         for c_id in range(len(self.p)):
    #             self.p[c_id][c_id] = np.min(self.p)
    #         for c_id in range(len(self.p)):
    #             self.p[c_id][c_id] = np.max(self.p)
    #         # print(f"client {self.client_id} p max {np.max(self.p)}")

    # 广播模块
    def broadcast(self):
        self.broadcaster.send(self.client_id, self.model)

    # 收到广播的回应，主要用来debug
    def response(self, sender_id):
        # logging.info("client[{:d}]:收到来自client[{:d}]的广播".format(self.client_id, sender_id))
        pass