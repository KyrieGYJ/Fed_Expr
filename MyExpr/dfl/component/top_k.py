import torch
import numpy as np
import heapq
import math

class TopKSelector(object):

    def __init__(self):
        self.recorder = None
        self.client_list = None
        self.select = None

    def use(self, indicator):
        description = "TopKSelector use strategy:{:s}"
        print(description.format(indicator))
        if indicator == "loss":
            self.select = self.top_k_by_loss
        elif indicator == "f1":
            self.select = self.top_k_by_f1

    def register_recoder(self, recorder):
        self.recorder = recorder

    def top_k_by_loss(self, train_X, train_Y, host_client_id):
        heap = []
        host_topology = self.recorder.topology_manager.get_symmetric_neighbor_list(host_client_id)
        for index in range(len(host_topology)):
            if host_topology[index] != 0 and index != host_client_id:
                neighbor = self.client_list[index]
                out = neighbor.model(train_X)
                loss = neighbor.criterion(out.squeeze(1), train_Y)
                heap.append([index, loss, out.squeeze(1)])
        # k 取 len(heap) * 0.8
        top_k = heapq.nlargest(math.floor(len(heap) * 0.8), heap, lambda x: x[1])
        return top_k

    def top_k_by_f1(self, train_X, train_Y, host_client):
        return None
