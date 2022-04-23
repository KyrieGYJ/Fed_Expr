import time
import heapq
import numpy as np
import copy
import random

from MyExpr.utils import generate_heatmap

class Broadcaster(object):

    def __init__(self, args):
        # neighbors_weight_dict
        self.args = args
        self.receive = None
        self.recorder = None
        self.strategy = None

        # 抽象方法
        self.send = None
        self.use(args.broadcaster_strategy)
        # 记录客户之间的通信频率
        self.broadcast_freq = None

    def register_recorder(self, recorder):
        self.recorder = recorder

    def initialize(self):
        # 通信频率矩阵
        self.broadcast_freq = np.zeros([self.args.client_num_in_total, self.args.client_num_in_total], dtype=np.float64)

    def use(self, strategy):
        description = "Broadcaster use strategy:{:s}"
        print(description.format(strategy))
        if strategy == "flood":
            self.send = self.flood
        elif strategy == "affinity_topK":
            self.send = self.affinity_topK
        elif strategy == "random":
            self.send = self.random

        self.strategy = strategy
        self.receive = self.receive_from_neighbors

    def flood(self, sender_id, model):
        client_dic = self.recorder.client_dic
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        for receiver_id in client_dic.keys():
            if topology[receiver_id] != 0 and receiver_id != sender_id:
                # print("{:d} 发送到 {:d}".format(sender_id, receiver_id))
                self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id], client_dic[sender_id].cache_keeper.broadcast_weight)

    def random(self, sender_id, model):
        client_dic = self.recorder.client_dic
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        if self.args.broadcast_K == -1:
            K = self.args.num_clients_per_dist
        else:
            K = int(self.args.broadcast_K * self.args.client_num_in_total)
        neighbor_list = []

        for neighbor_id in range(len(topology)):
            if topology[neighbor_id] != 0:
                neighbor_list.append(neighbor_id)

        np.random.seed(int(time.time()))  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(neighbor_list, K, replace=False)

        for receiver_id in client_indexes:
            self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id], client_dic[sender_id].cache_keeper.broadcast_weight)

    #######################################
    #         variants of affinity        #
    #######################################
    # 取topK
    def affinity_topK(self, sender_id, model, affinity_matrix=None):
        # 第一轮全发，避免后续出现差错

        # 根据上一轮接收到的neighbor模型，更新affinity矩阵，对矩阵聚类，并转发自身模型以及affinity权重到对应聚类上
        client_dic = self.recorder.client_dic
        sender = client_dic[sender_id]

        # 取出所有邻居
        candidate = []

        # 计算累计的affinity，如果本回合没收到，沿用上回合的
        # 这个策略可能有点粗暴，但是可以有效保证每次必然会发出固定数量，可能后续要改进
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        for neighbor_id in range(self.args.client_num_in_total):
            # 排除掉不存在的连接
            if topology[neighbor_id] == 0:
                continue
            candidate.append(neighbor_id)

        if self.args.broadcast_K == -1:
            K = self.args.num_clients_per_dist
        else:
            K = int(self.args.broadcast_K * self.args.client_num_in_total)

        # print(f"广播{K}个，num_clients_per_dist:{self.args.num_clients_per_dist}, total:{self.args.client_num_in_total}")
        random.shuffle(candidate)
        topK = heapq.nlargest(K, candidate, lambda x: sender.cache_keeper.p[x])

        # 转发
        # print(f"client {sender_id} are prone to be in the same distributions with {topK}")
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        for receiver_id in topK:
            # print(f"发送给client {receiver.client_id}")
            self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id],
                                        client_dic[sender_id].cache_keeper.broadcast_weight)
        # self_freq每次都加一，确保其全局最高 (不能这样做，这样会到导致中心太大，周围其他全部淡化)
        # self.broadcast_freq[sender_id][sender_id] += 1

    def get_freq_heatmap(self, path):
        if self.strategy == "flood":
            return
        epsilon = 1e-6
        n = self.args.client_num_in_total
        freq = copy.deepcopy(self.broadcast_freq)
        for c_id in range(n):
            freq[c_id][c_id] = freq.max()
        freq = (freq - freq.min()) / (freq.max() - freq.min() + epsilon)
        print("绘制通信频率热力图")
        generate_heatmap(freq, path)

    ###############
    #   接收方法   #
    ###############
    def receive_from_neighbors(self, sender_id, model, receiver_id, topology_weight, w = None):
        self.broadcast_freq[sender_id][receiver_id] += 1
        # 屏蔽自身消息
        if receiver_id == sender_id:
            return
        receiver = self.recorder.client_dic[receiver_id]
        # 调用receiver的方法，显示收到了某个client的数据。。（相当于钩子函数）
        receiver.response(sender_id)
        if model is not None:
            receiver.received_model_dict[sender_id] = model
        if topology_weight is not None:
            receiver.received_topology_weight_dict[sender_id] = topology_weight
        if w is not None:
            receiver.received_w_dict[sender_id] = w
