import torch
import numpy as np
import heapq
import math

class TopKSelector(object):

    def __init__(self, args):
        self.recorder = None
        self.select = None
        self.args = args

    def use(self, indicator):
        description = "TopKSelector use strategy:{:s}"
        print(description.format(indicator))
        if self.args.communication_wise == "epoch":
            if indicator == "loss":
                self.select = self.top_k_by_loss_epoch_wise
        else:
            if indicator == "loss":
                self.select = self.top_k_by_loss
            elif indicator == "f1_marco":
                self.select = self.top_k_by_f1_marco
            elif indicator == "f1_micro":
                self.select = self.top_k_by_f1_micro

    def register_recoder(self, recorder):
        self.recorder = recorder

    def top_k_by_loss_epoch_wise(self, host):
        heap = []
        host_id = host.client_id
        for index in host.neighbors_weight_dict:
             if index != host_id:
                total_loss = 0
                neighbor_model = host.neighbors_weight_dict[index]
                for idx, (val_X, val_Y) in enumerate(host.validation_loader):
                    val_X, val_Y = val_X.to(self.args.device), val_Y.to(self.args.device)
                    outputs = neighbor_model(val_X)
                    loss = host.criterion_CE(outputs, val_Y)
                    total_loss += loss
                heap.append([index, total_loss])
        top_k = heapq.nlargest(math.floor(10), heap, lambda x: x[1])
        return top_k

    def top_k_by_loss(self, train_X, train_Y, host):
        heap = []
        host_client_id = host.client_id
        # todo 这种是必然收到邻居模型的情况，可能要改成随机
        host_topology = self.recorder.topology_manager.get_symmetric_neighbor_list(host_client_id)
        for index in range(len(host_topology)):
            if host_topology[index] != 0 and index != host_client_id:
                neighbor_model = host.neighbors_weight_dict[index]
                outputs = neighbor_model(train_X)
                loss = host.criterion_CE(outputs, train_Y)
                heap.append([index, loss, outputs])
        # k 取 len(heap) * 0.8
        top_k = heapq.nlargest(math.floor(len(heap) * 0.8), heap, lambda x: x[1])
        return top_k

    def top_k_by_f1_marco(self, train_X, train_Y, host):
        heap = []
        host_client_id = host.client_id
        # todo 这种是必然收到邻居模型的情况，可能要改成随机
        host_topology = self.recorder.topology_manager.get_symmetric_neighbor_list(host_client_id)
        for index in range(len(host_topology)):
            if host_topology[index] != 0 and index != host_client_id:
                neighbor_model = host.neighbors_weight_dict[index]
                outputs = neighbor_model(train_X)
                pred = outputs.argmax(dim=1)
                pred_mask = torch.zeros(outputs.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
                pred_num = pred_mask.sum(0)  # 数据中每类的预测量
                targ_mask = torch.zeros(outputs.size()).scatter_(1, train_Y.data.cpu().view(-1, 1), 1.)
                targ_num = targ_mask.sum(0)  # 标签中每类的数量
                acc_mask = pred_mask * targ_mask
                acc_num = acc_mask.sum(0)  # 每类中预测正确的数量
                # 防止0/0产生nan
                epsilon = 1e-7
                recall = acc_num / (targ_num + epsilon)
                precision = acc_num / (pred_num + epsilon)
                f1 = 2 * recall * precision / (recall + precision + epsilon)
                f1_marco = f1.sum(0) / f1.size(0)
                heap.append([index, f1_marco, outputs])
        top_k = heapq.nlargest(math.floor(len(heap) * 0.8), heap, lambda x: x[1])
        return top_k

    def top_k_by_f1_micro(self, train_X, train_Y, host):
        heap = []
        host_client_id = host.client_id
        # todo 这种是必然收到邻居模型的情况，可能要改成随机
        host_topology = self.recorder.topology_manager.get_symmetric_neighbor_list(host_client_id)
        for index in range(len(host_topology)):
            if host_topology[index] != 0 and index != host_client_id:
                neighbor_model = host.neighbors_weight_dict[index]
                outputs = neighbor_model(train_X)
                pred = outputs.argmax(dim=1)
                pred_mask = torch.zeros(outputs.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
                targ_mask = torch.zeros(outputs.size()).scatter_(1, train_Y.data.cpu().view(-1, 1), 1.)
                acc_mask = pred_mask * targ_mask
                acc_num = acc_mask.sum(0)  # 每类中预测正确的数量
                # fp 为预测为positive，实际不是。pred去掉tp，剩下即为错误的positive预测，即为fp。
                fp = torch.clamp((pred_mask - targ_mask), 0.0).sum(0)
                # fn 为预测为negative，实际不是。targ去掉tp，剩下的即为没预测到的positive样本，即为fn。
                fn = torch.clamp((targ_mask - pred_mask), 0.0).sum(0)
                acc_sum = acc_num.sum(0)
                fp_sum = fp.sum(0)
                fn_sum = fn.sum(0)
                # 防止0/0产生nan
                epsilon = 1e-7
                precision_sum = acc_sum / (acc_sum + fp_sum + epsilon)
                recall_sum = acc_sum / (acc_sum + fn_sum + epsilon)
                f1_micro = 2 * recall_sum * precision_sum / (recall_sum + precision_sum + epsilon)
                heap.append([index, f1_micro, outputs])
        top_k = heapq.nlargest(math.floor(len(heap) * 0.8), heap, lambda x: x[1])
        return top_k

