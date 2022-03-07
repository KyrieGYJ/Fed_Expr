import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import heapq
import math
import numpy as np


class ClientTEST(object):
    def __init__(self, model, client_id, streaming_data, topology_manager, iteration_number,
                 learning_rate, batch_size, weight_decay, latency, b_symmetric):
        # logging.info("streaming_data = %s" % streaming_data)

        # Since we use logistic regression, the model size is small.
        # Thus, independent model is created each client.
        self.model = model

        self.b_symmetric = b_symmetric
        self.topology_manager = topology_manager
        self.id = client_id  # integer
        self.streaming_data = streaming_data

        if self.b_symmetric:
            self.topology = topology_manager.get_symmetric_neighbor_list(client_id)
        else:
            self.topology = topology_manager.get_asymmetric_neighbor_list(client_id)
        # print(self.topology)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.BCELoss()

        self.learning_rate = learning_rate
        self.iteration_number = iteration_number

        # TODO:
        self.latency = random.uniform(0, latency)

        self.batch_size = batch_size
        self.loss_in_each_iteration = []
        self.record = []

        self.last_loss = None
        self.last_outputs = None

        # neighbors_weight_dict
        self.neighbors_weight_dict = dict()
        self.neighbors_topo_weight_dict = dict()

    def train(self, iteration_id):
        self.optimizer.zero_grad()

        if iteration_id >= self.iteration_number:
            iteration_id = iteration_id % self.iteration_number

        train_x = torch.from_numpy(np.asarray(self.streaming_data['x'][iteration_id], dtype=np.float32))
        train_y = torch.from_numpy(np.asarray(self.streaming_data['y'][iteration_id], dtype=np.float32))
        # train_x = torch.from_numpy(self.streaming_data['x'][iteration_id]).float()
        # print(train_x)
        # train_y = torch.FloatTensor([self.streaming_data['y'][iteration_id]])
        # outputs = [100 * 1], batch_size = 100, output_dim = 1
        outputs = self.model(train_x)
        # print(train_y)
        # loss = self.criterion(outputs, train_y)
        loss = self.criterion(outputs.squeeze(1), train_y)
        # print(loss)

        # 梯度
        # grads_z = torch.autograd.grad(loss, self.model.parameters())

        # 准确率
        pred = outputs.ge(0.5).float()
        correct = (pred.squeeze(1) == train_y).sum().item()

        # 更新model-cache, 即z_t
        # for x_paras, g_z in zip(list(self.model.parameters()), grads_z):
        #     temp = g_z.data.mul(0 - self.learning_rate)
        #     x_paras.data.add_(temp)

        # self.loss_in_each_iteration.append(loss)
        self.record.append((correct, len(train_y)))
        self.last_loss = loss
        self.last_outputs = outputs

    def get_regret(self):
        return self.loss_in_each_iteration

    def get_record(self):
        return self.record

    # simulation
    def send_local_gradient_to_neighbor(self, client_list):
        for index in range(len(self.topology)):
            if self.topology[index] != 0 and index != self.id:
                client = client_list[index]
                client.receive_neighbor_gradients(self.id, self.model, self.topology[index])

    def receive_neighbor_gradients(self, client_id, model, topo_weight):
        self.neighbors_weight_dict[client_id] = model
        self.neighbors_topo_weight_dict[client_id] = topo_weight

    def mutual_update(self, top_k):
        criterion_KLD = nn.KLDivLoss(reduction='batchmean')
        # 聚合top_k的总KL散度
        total_loss = self.last_loss
        outputs = self.last_outputs
        KL_loss = 0
        for _ , (index, loss, out) in enumerate(top_k):
            KL_loss += criterion_KLD(torch.log(out), outputs.squeeze(1)).item()
            # print(outputs.shape, out.shape)
            # print("outputs:", outputs, "out:", out, "KL_loss: ", KL_loss)
        total_loss += KL_loss / (len(top_k))
        print("KL_loss: ", KL_loss)

        grads_z = torch.autograd.grad(total_loss, self.model.parameters())

        for x_paras, g_z in zip(list(self.model.parameters()), grads_z):
            temp = g_z.data.mul(0 - self.learning_rate)
            x_paras.data.add_(temp)

        self.loss_in_each_iteration.append(total_loss.detach().numpy())

    def top_k_by_loss(self, client_list, iteration_id):
        heap = []
        train_x = torch.from_numpy(np.asarray(self.streaming_data['x'][iteration_id], dtype=np.float32))
        train_y = torch.from_numpy(np.asarray(self.streaming_data['y'][iteration_id], dtype=np.float32))
        for index in range(len(self.topology)):
            if self.topology[index] != 0 and index != self.id:
                neighbor = client_list[index]
                out = neighbor.model(train_x)
                loss = neighbor.criterion(out.squeeze(1), train_y)
                heap.append([index, loss, out.squeeze(1)])
        # k 取 len(heap) * 0.8
        top_k = heapq.nlargest(math.floor(len(heap) * 0.8), heap, lambda x: x[1])
        return top_k

    def clear(self):
        self.loss_in_each_iteration.clear()
        self.record.clear()