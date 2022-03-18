from MyExpr.utils import cal_w
import heapq

class Broadcaster(object):

    def __init__(self):
        # neighbors_weight_dict
        self.receive = None
        self.send = None
        self.recorder = None
        self.strategy = None
        # affinity metrics策略用用到
        self.p = None
        self.w = None

    def register_recorder(self, recorder):
        self.recorder = recorder

    def initialize(self):
        if self.strategy == "affinity":
            # 初始化权重矩阵，不连通的边用-1填充
            client_dic = self.recorder.client_dic
            self.p = [[1 for _ in client_dic] for _ in client_dic]
            self.w = [[1 for _ in client_dic] for _ in client_dic]
            for c_id in client_dic:
                topology = self.recorder.topology_manager.get_symmetric_neighbor_list(c_id)
                for neighbor_id in client_dic:
                    if topology[neighbor_id] == 0 or neighbor_id == c_id:
                        self.p[c_id][neighbor_id] = -1
                        self.w[c_id][neighbor_id] = -1

    def use(self, strategy):
        description = "Broadcaster use strategy:{:s}"
        print(description.format(strategy))
        if strategy == "flood":
            self.send = self.flood
        elif strategy == "affinity":
            self.send = self.affinity
        self.strategy = strategy
        self.receive = self.receive_from_neighbors

    # todo 后续要利用上topology_weight（类似dfl论文里的参考pagerank）
    def flood(self, sender_id, model):
        client_dic = self.recorder.client_dic
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        num = 0
        for receiver_id in client_dic.keys():
            if topology[receiver_id] != 0 and receiver_id != sender_id:
                # print("{:d} 发送到 {:d}".format(sender_id, receiver_id))
                self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id])
                num += 1
        print("client {} has {} neighbors".format(sender_id, num))

    def affinity(self, sender_id, model):
        topology = self.recorder.topology_manager.get_symmetric_neighbor_list(sender_id)
        # 第一轮全发，避免后续出现差错
        if self.recorder.rounds == 0:
            self.flood(sender_id, model)
            return
        # 选15个广播
        client_dic = self.recorder.client_dic
        new_w = cal_w(client_dic[sender_id], self.recorder.args)
        sender_p_metric = self.p[sender_id]
        sender_w_metric = self.w[sender_id]
        candidate = []
        # 计算累计的affinity，如果本回合没收到，沿用上回合的
        # 这个策略可能有点粗暴，但是可以有效保证每次必然会发出固定数量，可能后续要改进
        for neighbor_id in range(len(sender_p_metric)):
            # 排除掉不存在的连接
            if neighbor_id == sender_id or sender_p_metric[neighbor_id] == -1:
                continue
            # 上一轮收到了neighbor_id的模型
            if neighbor_id in new_w:
                sender_p_metric[neighbor_id] += new_w[neighbor_id]
                sender_w_metric[neighbor_id] = new_w[neighbor_id]
            else:
                sender_p_metric[neighbor_id] += sender_w_metric[neighbor_id]
            candidate.append(neighbor_id)
        top_15 = heapq.nlargest(15, candidate, lambda x: sender_p_metric[x])
        for receiver_id in top_15:
            self.receive_from_neighbors(sender_id, model, receiver_id, topology[receiver_id])

    ###############
    #   接收方法   #
    ###############
    def receive_from_neighbors(self, sender_id, model, receiver_id, topology_weight):
        receiver = self.recorder.client_dic[receiver_id]
        # 调用receiver的方法，显示收到了某个client的数据。。（相当于钩子函数）
        receiver.response(sender_id)
        receiver.received_model_dict[sender_id] = model
        receiver.received_topology_weight_dict[sender_id] = topology_weight
