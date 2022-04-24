import torch
import torch.nn as nn
import numpy as np
import copy
from opacus import PrivacyEngine


from .cache_keeper import keeper
from .logger import logger


class Client(object):
    def __init__(self, model, client_id, args, data, topK_selector, recorder, broadcaster):
        self.args = args
        self.client_id = client_id
        self.data = data
        self.log_condition = client_id == 0
        self.logger = logger(self, "client")

        total_benign_num = args.client_num_in_total - args.malignant_num
        train_loader = data.train_loader[client_id] if client_id < total_benign_num else None
        validation_loader = data.validation_loader[client_id] if client_id < total_benign_num else None
        test_loader = data.test_loader[client_id] if client_id < total_benign_num else None

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

        self.train_set = data.train_set[client_id] if client_id < total_benign_num else None
        self.test_set = data.test_set[client_id] if client_id < total_benign_num else None
        self.validation_set = data.validation_set[client_id] if client_id < total_benign_num else None

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

    def local_test(self):
        total_loss, total_correct = 0., 0
        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            for _, (test_X, test_Y) in enumerate(self.test_loader):
                test_X, test_Y = test_X.to(self.args.device), test_Y.to(self.args.device)
                loss, correct = self.test(test_X, test_Y)
                total_loss += loss.item()
                total_correct += correct
        return total_loss, total_correct

    ##############
    #  an epoch  #
    ##############
    def local_train(self):
        self.model.train()
        self.model.to(self.args.device)
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

    def weighted_model_interpolation_update3(self):
        local_weight = self.cache_keeper.mutual_update_weight3[self.client_id]
        total_weight = local_weight
        for x_paras in self.model.parameters():
            x_paras.data.mul_(local_weight)

        # self.logger.log_with_name(f"received: {self.received_model_dict.keys()}", self.log_condition)

        for neighbor_id in self.received_model_dict.keys():
            neighbor_model = self.received_model_dict[neighbor_id]
            neighbor_weight = self.cache_keeper.mutual_update_weight3[neighbor_id]
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
        self.topK_neighbor = {}
        # todo 改这里
        for c_id, loss in topK:
            self.topK_neighbor[c_id] = self.received_model_dict[c_id]
        # print("client {} 本轮topK选择了{}个模型".format(self.client_id, len(self.received_model_dict)))

    # 广播模块
    def broadcast(self):
        self.broadcaster.send(self.client_id, self.model)

    # 收到广播的回应，主要用来debug
    def response(self, sender_id):
        # logging.info("client[{:d}]:收到来自client[{:d}]的广播".format(self.client_id, sender_id))
        pass