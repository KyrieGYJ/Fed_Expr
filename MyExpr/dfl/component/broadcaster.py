
import time
from MyExpr.utils import cal_raw_w
import heapq
import numpy as np
import torch
import copy
import os

from MyExpr.utils import generate_heatmap
from MyExpr.utils import get_adjacency_matrix

from sklearn.cluster import AgglomerativeClustering, SpectralClustering

class Broadcaster(object):

    def __init__(self, args):
        # neighbors_weight_dict
        self.args = args
        self.receive = None
        self.recorder = None
        self.strategy = None
        # affinity metrics策略用用到
        self.p = None
        self.w = None

        # 抽象方法
        self.send = None
        self.get_w_heatmap = None

        self.use(args.broadcaster_strategy)

        # 记录客户之间的通信频率
        self.broadcast_freq = None

    def register_recorder(self, recorder):
        self.recorder = recorder

    def initialize(self):
        if self.strategy == "affinity":
            # 初始化权重矩阵，不连通的边用-1填充
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
                    # if topology[neighbor_id] == 0:
                    #     self.p[c_id][neighbor_id] = torch.tensor(0.)
                    #     self.w[c_id][neighbor_id] = torch.tensor(0.)
                    # elif neighbor_id == c_id:
                    #     self.p[c_id][neighbor_id] = torch.tensor(1. * client_initial_self_weight)
                    #     self.w[c_id][neighbor_id] = torch.tensor(1. * client_initial_self_weight)
                    if topology[neighbor_id] == 0:
                        self.p[c_id][neighbor_id] = 0.
                        self.w[neighbor_id] = 0.
                    elif neighbor_id == c_id:
                        self.p[c_id][neighbor_id] = 1. * client_initial_self_weight
                        self.w[neighbor_id] = 1. * client_initial_self_weight

        # 通信频率矩阵
        self.broadcast_freq = np.zeros([self.args.client_num_in_total, self.args.client_num_in_total], dtype=np.float64)

    def use(self, strategy):
        description = "Broadcaster use strategy:{:s}"
        print(description.format(strategy))
        if strategy == "flood":
            self.send = self.flood
            self.get_w_heatmap = self.pass_heatmap
        elif strategy == "affinity":
            self.send = self.affinity
            self.get_w_heatmap = self.get_p_heatmap
        elif strategy == "affinity_cluster":
            self.send = self.affinity_cluster
            self.get_w_heatmap = self.get_clients_affinity_heatmap
        elif strategy == "affinity_topK":
            self.send = self.affinity_topK
            self.get_w_heatmap = self.get_clients_affinity_heatmap
        elif strategy == "affinity_baseline":
            self.send = self.affinity_baseline
            self.get_w_heatmap = self.get_clients_affinity_heatmap
        elif strategy == "random":
            self.send = self.random
            self.get_w_heatmap = self.pass_heatmap

        self.strategy = strategy
        self.receive = self.receive_from_neighbors

    # todo 后续可能要利用上topology_weight（类似dfl论文里的参考pagerank）
    def flood(self, sender_id, model):
        client_dic = self.recorder.client_dic
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        # num = 0
        for receiver_id in client_dic.keys():
            if topology[receiver_id] != 0 and receiver_id != sender_id:
                # print("{:d} 发送到 {:d}".format(sender_id, receiver_id))
                self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id], client_dic[sender_id].broadcast_w)
                # num += 1
        # print("client {} has {} neighbors".format(sender_id, num))

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
            self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id], client_dic[sender_id].broadcast_w)

    #######################################
    #         variants of affinity        #
    # each client hold an affintiy matrix #
    #######################################
    # 对affinity采用谱聚类（不合适，存在bug）
    def affinity_cluster(self, sender_id, model, affinity_matrix=None):
        args = self.args

        # 第一轮全发，避免后续出现差错
        if affinity_matrix is None and self.recorder.rounds == 0:
            self.flood(sender_id, model)
            return

        # 根据上一轮接收到的neighbor模型，更新affinity矩阵，对矩阵聚类，并转发自身模型以及affinity权重到对应聚类上
        client_dic = self.recorder.client_dic
        sender = client_dic[sender_id]

        if affinity_matrix is None:
            sender.update_broadcast_weight()
            sender.update_p()
            # 根据p生成affinity矩阵
            affinity_matrix = get_adjacency_matrix(sender)

        sender.affinity_matrix = affinity_matrix
        # 检查矩阵是否正常
        # for c_id in range(args.client_num_in_total):
        #     if affinity_matrix[c_id][c_id] != np.max(affinity_matrix):
        #         print("AFFINITY ERROR")
        # print(affinity_matrix)
        # 聚类
        clustering = SpectralClustering(n_clusters=args.num_distributions,
                                        affinity='precomputed',
                                        n_init=10, random_state=args.seed,
                                        assign_labels='discretize')
        federation_labels = clustering.fit_predict(affinity_matrix)
        # 取出属于同一个聚类的client
        clients_per_federation = [[] for _ in range(args.num_distributions)]
        sender_label = None
        for ix, label in enumerate(federation_labels):
            if ix == sender_id:
                sender_label = label
            clients_per_federation[label].append(
                client_dic[ix])
        # 转发
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        # print(f"client {sender_id} are in the same distribution of clients: {[c.client_id for c in self.clients_per_federation[sender_label]]}，权重w={sender.w}")
        print(f"client {sender_id} are in the same distribution of clients: {[c.client_id for c in clients_per_federation[sender_label]]}")
        # print(f"affinity matrix minimum: {np.min(sender.affinity_matrix)}")
        # print(f"p matrix is {sender.p} \n ===================")
        # print(f"affinity matrix is {sender.affinity_matrix}")
        for receiver in clients_per_federation[sender_label]:
            # print(f"发送给client {receiver.client_id}")
            if receiver.client_id == sender_id:
                continue
            self.receive_from_neighbors(sender_id, model, receiver.client_id, topology[receiver.client_id], sender.broadcast_w)

    # 取topK
    def affinity_topK(self, sender_id, model, affinity_matrix=None):
        # 第一轮全发，避免后续出现差错
        if affinity_matrix is None and self.recorder.rounds == 0:
            self.flood(sender_id, model)
            return

        # 根据上一轮接收到的neighbor模型，更新affinity矩阵，对矩阵聚类，并转发自身模型以及affinity权重到对应聚类上
        client_dic = self.recorder.client_dic
        sender = client_dic[sender_id]

        if affinity_matrix is None:
            # （自身权重取max）
            sender.update_broadcast_weight()
            sender.update_p()
            # 根据p生成affinity矩阵(正规化)
            affinity_matrix = get_adjacency_matrix(sender)
            # print(f"client {sender_id} affinity max={np.max(affinity_matrix)}， affinity min {np.min(affinity_matrix)}")

        sender.affinity_matrix = affinity_matrix

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

        # topK = heapq.nlargest(self.args.num_clients_per_dist, candidate, lambda x: sender.affinity_matrix[sender_id][x])
        if self.args.broadcast_K == -1:
            K = self.args.num_clients_per_dist
        else:
            K = int(self.args.broadcast_K * self.args.client_num_in_total)
        # print(f"广播{K}个，num_clients_per_dist:{self.args.num_clients_per_dist}, total:{self.args.client_num_in_total}")
        topK = heapq.nlargest(K, candidate, lambda x: sender.affinity_matrix[sender_id][x])

        # 转发
        # print(f"client {sender_id} are prone to be in the same distributions with {topK}")
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        for receiver_id in topK:
            # print(f"发送给client {receiver.client_id}")
            self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id], sender.broadcast_w)

    def affinity_baseline(self, sender_id, model, affinity_matrix=None):
        args = self.args

        # 第一轮全发，避免后续出现差错
        if affinity_matrix is None and self.recorder.rounds == 0:
            self.flood(sender_id, model)
            return

        # 根据真实聚类计算w，更新p矩阵和affinity矩阵，
        client_dic = self.recorder.client_dic
        sender = client_dic[sender_id]

        if affinity_matrix is None:
            sender.update_broadcast_weight()
            sender.update_p()
            # 根据p生成affinity矩阵
            affinity_matrix = get_adjacency_matrix(sender)

        sender.affinity_matrix = affinity_matrix

        # 获取真实聚类
        dist_client_dict = self.recorder.data.dist_client_dict
        client_dist_dict = self.recorder.data.client_dist_dict
        sender_dist_id = client_dist_dict[sender_id]
        sender_dist = dist_client_dict[sender_dist_id]

        # 转发
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        print(f"client {sender_id} are in the same distribution of clients: {sender_dist}")
        for receiver_id in sender_dist:
            if receiver_id == sender_id:
                continue
            self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id], sender.broadcast_w)

    def affinity(self, sender_id, model):
        # 不接收邻居的广播权重，只按照自身的计算
        # affinity矩阵是整合所有client自身的广播权重生成的
        # 第一轮全发，避免后续出现差错
        if self.recorder.rounds == 0:
            self.flood(sender_id, model)
            return
        # 选15个广播
        client_dic = self.recorder.client_dic
        new_w = cal_raw_w(client_dic[sender_id], self.recorder.args)
        sender_p_metric = self.p[sender_id]
        sender_w_metric = self.w[sender_id]
        candidate = []

        # 计算累计的affinity，如果本回合没收到，沿用上回合的
        # 这个策略可能有点粗暴，但是可以有效保证每次必然会发出固定数量，可能后续要改进
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        for neighbor_id in range(len(sender_p_metric)):
            # 排除掉不存在的连接
            # if neighbor_id == sender_id or sender_p_metric[neighbor_id] == -1:
            if neighbor_id == sender_id or topology[neighbor_id] == 0:
                continue
            # 上一轮收到了neighbor_id的模型
            if neighbor_id in new_w:
                sender_p_metric[neighbor_id] += new_w[neighbor_id]
                # 更新权重w的缓存
                sender_w_metric[neighbor_id] = new_w[neighbor_id]
            else:
                # 本轮没收到则沿用上回的w更新p矩阵
                sender_p_metric[neighbor_id] += sender_w_metric[neighbor_id]
            candidate.append(neighbor_id)
        # 改成广播到对应聚类上
        top_15 = heapq.nlargest(15, candidate, lambda x: sender_p_metric[x])
        # losses = [sender_p_metric[idx] for idx in top_15]
        # print("client {} top15 is {}, their loss are {} respectively and new_w are {}".format(sender_id, top_15, losses, new_w))
        for receiver_id in top_15:
            self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id])

    # 打印broadcaster上的p矩阵（仅对affinity()有意义）
    def get_p_heatmap(self, path):
        if self.strategy == "flood":
            return
        epsilon = 1e-6
        n = self.args.client_num_in_total
        p_list = np.zeros([n, n], dtype=np.float64)
        # print(self.p)
        for c_id_from in range(n):
            for c_id_to in range(n):
                p_list[c_id_from][c_id_to] = self.p[c_id_from][c_id_to].item()
                # print(type(p_list[c_id_from][c_id_to]))
        p_list = (p_list - p_list.min()) / (p_list.max() - p_list.min() + epsilon)
        # print(p_list)
        print("绘制权重热力图")
        generate_heatmap(p_list, path)

    # todo 这两个方法位置不合逻辑
    def get_clients_affinity_heatmap(self, path):
        print("绘制affinity热力图")
        for c_id in range(self.args.client_num_in_total):
            p_list = self.recorder.client_dic[c_id].affinity_matrix
            if p_list is None or p_list == []:
                continue
            if not os.path.exists(path):
                os.makedirs(path)
            generate_heatmap(p_list, f"{path}/client_{c_id}")

    def get_clients_p_heatmap(self, path):
        print("绘制p矩阵热力图")
        for c_id in range(self.args.client_num_in_total):
            p_list = self.recorder.client_dic[c_id].p
            if p_list is None or p_list == []:
                continue
            if not os.path.exists(path):
                os.makedirs(path)
            generate_heatmap(p_list, f"{path}/client_{c_id}")

    def pass_heatmap(self, path):
        pass

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


