import copy
import numpy as np
from MyExpr.utils import calc_delta_loss, calc_eval, compute_parameter_difference, eval


class keeper(object):
    """
    主要是解耦权重模块，不然client太重了
    """

    def __init__(self, host):
        self.host = host
        self.recorder = host.recorder
        self.args = self.recorder.args

        # 接收缓存
        self.model_memory = {}
        self.topology_weight_memory = {}
        self.broadcast_weight_memory = {}
        # 权重缓存（用于计算通信权重，对于本轮没收到的模型复用历史记录）
        self.delta_loss_memory = {}
        self.model_dif_memory = {}

        # 权重模块
        self.delta_loss_dict = {}
        self.delta_loss_prior_dict = {}
        self.raw_eval_loss_dict = {}
        self.raw_eval_acc_dict = {}
        self.dif_list = []
        self.dif_list_prior = [] # 采用local_model prior to its current state
        self.broadcast_weight = []

        self.mutual_update_weight = [] # 采用delta_loss计算
        self.mutual_update_weight2 = [] # 采用raw_loss计算
        self.mutual_update_weight3 = [] # 采用delta_loss计算，但是加入自因子
        self.mutual_update_weight4 = [] # 采用raw_acc计算
        self.mutual_update_weight5 = [] # 采用local_model prior to its current state
        self.sigma = 0.1 # 防止model_dif最大的delta loss丢失

        # local_train前的模型缓存
        self.last_local_model = None
        self.p = None # 原始权重矩阵
        self.affinity_matrix = None # 调整后的权重矩阵（p和affinity应该是等价的才行）

        self.update_uw = None

    def initialize(self):
        self.p = np.zeros([self.args.client_num_in_total, self.args.client_num_in_total],
                         dtype=np.float64)
        if self.args.trainer_strategy == "weighted_model_interpolation1":
            self.update_uw = self.update_uw1
        elif self.args.trainer_strategy == "weighted_model_interpolation2":
            self.update_uw = self.update_uw2
        elif self.args.trainer_strategy == "weighted_model_interpolation3":
            self.update_uw = self.update_uw3
        elif self.args.trainer_strategy == "weighted_model_interpolation4":
            self.update_uw = self.update_uw4
        elif self.args.trainer_strategy == "weighted_model_interpolation5":
            self.update_uw = self.update_uw5

    def update_weight(self):
        self.update_delta_loss()
        self.dif_list = self.update_model_difference()
        self.dif_list_prior = self.update_model_difference(local_model=self.last_local_model)

        self.update_uw()
        self.update_bw()

    def update_affinity_map(self):
        self.update_p()
        self.update_affinity_matrix()

    def update_delta_loss(self):
        self.raw_eval_loss_dict, self.raw_eval_acc_dict = calc_eval(self.host, self.model_memory, self.args)
        local_loss = self.raw_eval_loss_dict[self.host.client_id]

        # 使用local_train后的本地模型作为基准模型 delta_loss = L_i(t) - L_n(t)
        self.delta_loss_dict = {}
        # self.delta_loss = [0. for _ in range(self.args.client_num_in_total)]
        for c_id in self.raw_eval_loss_dict:
            self.delta_loss_dict[c_id] = local_loss - self.raw_eval_loss_dict[c_id]

        # self.delta_loss = calc_delta_loss(self.host, self.model_memory, self.args)
        # print(f"cache keeper: delta_loss1:{self.raw_eval_loss} \n delta_loss:{self.delta_loss}")

        # 使用local_train前的本地模型作为基准模型 delta_loss_prior = L_i(t-1) - L_n(t)
        local_loss_prior, _ = eval(self.last_local_model, self.host, self.args)
        self.delta_loss_prior_dict = {}
        # self.delta_loss_prior = [0. for _ in range(self.args.client_num_in_total)]
        for c_id in self.raw_eval_loss_dict:
            self.delta_loss_prior_dict[c_id] = local_loss_prior - self.raw_eval_loss_dict[c_id]
        print(f"cache keeper: client {self.host.client_id} keys-delta_loss_dict:{self.delta_loss_dict.keys()} \n "
              f"keys-delta_loss_prior_dict:{self.delta_loss_prior_dict.keys()}")

    def update_model_difference(self, local_model=None, norm=True):
        # model_difference_dict = cal_model_dif(self.host, self.args, norm_type="l2_root")
        epsilon = 1e-9
        model_difference_dict = {}
        mode_name = "local model" if local_model is None else "local prior model"
        local_model = self.host.model if local_model is None else local_model

        # received_dict = self.host.received_model_dict
        received_model_dict = self.model_memory
        for c_id, model in received_model_dict.items():
            dif = compute_parameter_difference(local_model, model, norm="l2")
            model_difference_dict[c_id] = dif
        # 添加自身model_dif
        model_difference_dict[self.host.client_id] = compute_parameter_difference(local_model, self.host.model, norm="l2")
        dif_list = [0 for _ in range(self.args.client_num_in_total)]

        for c_id, dif in model_difference_dict.items():
            dif_list[c_id] = dif

        # 防止0项干扰norm计算
        for c_id in range(self.args.client_num_in_total):
            if c_id not in received_model_dict:
                dif_list[c_id] = np.max(dif_list)
        for c_id in range(self.args.client_num_in_total):
            if c_id not in received_model_dict:
                dif_list[c_id] = np.min(dif_list)

        # print(f"cache_keeper: client {self.host.client_id} model_dif[{mode_name}] - max:{np.max(dif_list)} min:{np.min(dif_list)}")

        dif_list = np.array(dif_list)
        # print(f"cache_keeper: client {self.host.client_id} raw model_dif:{dif_list}, max:{np.max(dif_list)}, min:{np.min(dif_list)}")
        if norm:
            dif_list = (dif_list - np.min(dif_list)) / (np.max(dif_list) - np.min(dif_list) + epsilon)  # norm
        print(f"cache keeper: client {self.host.client_id} keys-model_difference_dict:{model_difference_dict.keys()}")
        # print(f"cache_keeper: client {self.host.client_id} norm model_dif:{dif_list}, max:{np.max(dif_list)}, min:{np.min(dif_list)}")
        return dif_list

    def check_update_weight(self, model_dif_adjust=True):
        self.update_uw1(model_dif_adjust)
        self.update_uw2(model_dif_adjust)
        self.update_uw3(model_dif_adjust)
        self.update_uw4(model_dif_adjust)
        self.update_uw5(model_dif_adjust)

    # todo 直接deepcopylist来用
    def update_uw1(self, model_dif_adjust=True):
        delta_loss = self.delta_loss_dict
        new_update_w_list = []

        epsilon = 1e-9
        for i in range(self.args.client_num_in_total):
            # relu
            if i in delta_loss and delta_loss[i] >= 0:
                new_update_w_list.append(copy.deepcopy(delta_loss[i]))
            else:
                new_update_w_list.append(0)

        if model_dif_adjust:
            new_update_w_list *= (1 - self.dif_list + self.sigma) # 和模型差距成反比

        # 更新权重需要norm，邻居模型贡献的总权重为1
        norm_factor = max(np.sum(new_update_w_list), epsilon)
        new_update_w_list = np.array(new_update_w_list) / norm_factor

        # print(f"client {self.host.client_id} new_update_w_list : {new_update_w_list}")
        self.mutual_update_weight = new_update_w_list
        print(f"cache keeper: client{self.host.client_id} new_update_w_list {new_update_w_list}")

    def update_uw2(self, model_dif_adjust=True):
        raw_eval_loss = self.raw_eval_loss_dict
        new_update_w_list = []

        epsilon = 1e-9
        for i in range(self.args.client_num_in_total):
            # relu
            if i in raw_eval_loss and self.delta_loss_dict[i] >= 0:
                new_update_w_list.append(1 / (copy.deepcopy(raw_eval_loss[i]) + epsilon))
            else:
                new_update_w_list.append(0)

        if model_dif_adjust:
            new_update_w_list *= (1 - self.dif_list + self.sigma)  # 和模型差距成反比

        # 更新权重需要norm，邻居模型贡献的总权重为1
        norm_factor = max(np.sum(new_update_w_list), epsilon)
        new_update_w_list = np.array(new_update_w_list) / norm_factor

        # print(f"client {self.host.client_id} new_update_w_list : {new_update_w_list}")
        self.mutual_update_weight2 = new_update_w_list
        print(f"cache keeper: client{self.host.client_id} new_update_w2_list {new_update_w_list}")

    def update_uw3(self, model_dif_adjust=True):
        delta_loss = self.delta_loss_dict
        new_update_w_list = []

        epsilon = 1e-9
        for i in range(self.args.client_num_in_total):
            # relu 忽略负效果模型
            if i in delta_loss and delta_loss[i] >= 0:
                new_update_w_list.append(copy.deepcopy(delta_loss[i] + self.sigma))
            else:
                new_update_w_list.append(0)
        new_update_w_list = np.array(new_update_w_list)

        if model_dif_adjust:
            # delta_loss + sigma: 防止因为delta_loss为0而在聚合时忽略自身
            # 1 - model_dif + sigma: 防止model_dif最大者被忽略
            new_update_w_list = new_update_w_list * (1 - self.dif_list + self.sigma)  # 和模型差距成反比

        # 更新权重需要norm，邻居模型贡献的总权重为1
        norm_factor = max(np.sum(new_update_w_list), epsilon)
        new_update_w_list = np.array(new_update_w_list) / norm_factor
        self.mutual_update_weight3 = new_update_w_list
        print(f"cache keeper: client{self.host.client_id} new_update_w3_list {new_update_w_list}")

    def update_uw4(self, model_dif_adjust=True):
        raw_acc = self.raw_eval_acc_dict
        new_update_w_list = []

        epsilon = 1e-9
        for i in range(self.args.client_num_in_total):
            # relu 忽略负效果模型
            if i in raw_acc and self.delta_loss_dict[i] >= 0:
                new_update_w_list.append(copy.deepcopy(raw_acc[i]))
            else:
                new_update_w_list.append(0)
        new_update_w_list = np.array(new_update_w_list)

        if model_dif_adjust:
            new_update_w_list = new_update_w_list * (1 - self.dif_list + self.sigma)  # 和模型差距成反比

        # 更新权重需要norm，邻居模型贡献的总权重为1
        norm_factor = max(np.sum(new_update_w_list), epsilon)
        new_update_w_list = np.array(new_update_w_list) / norm_factor
        self.mutual_update_weight4 = new_update_w_list
        print(f"cache keeper: client{self.host.client_id} new_update_w4_list {new_update_w_list}")

    def update_uw5(self, model_dif_adjust=True):
        delta_loss = self.delta_loss_prior_dict
        new_update_w_list = []

        epsilon = 1e-9
        for i in range(self.args.client_num_in_total):
            # relu 忽略负效果模型
            if i in delta_loss and delta_loss[i] >= 0:
                new_update_w_list.append(copy.deepcopy(delta_loss[i]))
            else:
                new_update_w_list.append(0)
        new_update_w_list = np.array(new_update_w_list)
        # print(f"cache keeper: before adjustment client{self.host.client_id} delta_loss - max:{np.max(new_update_w_list)}, true min:{np.partition(new_update_w_list, -2)[-2]}, sum: {np.sum(new_update_w_list)}")
        if model_dif_adjust:
            # 1 - model_dif + sigma: 防止model_dif最大者被忽略
            new_update_w_list = new_update_w_list * (1 - self.dif_list_prior + self.sigma)  # 和模型差距成反比

        # 更新权重需要norm，邻居模型贡献的总权重为1
        norm_factor = max(np.sum(new_update_w_list), epsilon)
        new_update_w_list = np.array(new_update_w_list) / norm_factor
        self.mutual_update_weight5 = new_update_w_list
        # print(f"cache keeper: client{self.host.client_id} new_update_w5_list {new_update_w_list}")
        # print(f"cache keeper: after adjustment client{self.host.client_id} new_update_w5_list - max:{np.max(new_update_w_list)}, true min:{np.partition(new_update_w_list, -2)[-2]}, sum: {np.sum(new_update_w_list)}")

    def update_bw(self, balanced=False, model_dif_adjust=True):
        delta_loss = self.delta_loss_dict
        new_broadcast_w_list = []

        for i in range(self.args.client_num_in_total):
            if i in delta_loss:
                new_broadcast_w_list.append(copy.deepcopy(delta_loss[i]))
            else:
                new_broadcast_w_list.append(0)

        # print(f"before model_dif_adjust client {self.host.client_id} new_broadcast_w_list : {new_broadcast_w_list}")

        if model_dif_adjust:
            new_broadcast_w_list *= (1 - self.dif_list + self.sigma) # 和模型差距成反比

        # print(f"model_dif: {self.dif_list}")
        # print(f"1 - model_dif: {(1 - self.dif_list + self.sigma)}")
        new_broadcast_w_list = np.array(new_broadcast_w_list)
        # print(f"after model_dif_adjust client {self.host.client_id} new_broadcast_w_list : {new_broadcast_w_list}")
        if balanced:
            new_broadcast_w_list /= len(self.host.validation_set)

        self.broadcast_weight = new_broadcast_w_list
        # print(f"cache keeper: client{self.host.client_id} new_broadcast_w_list {new_broadcast_w_list}")

    def update_p(self, self_max=True):
        # if self.p is None:
        #     self.initialize()

        self.p[self.host.client_id] += self.broadcast_weight

        for neighbor_id in self.broadcast_weight_memory:
            self.p[neighbor_id] += self.broadcast_weight_memory[neighbor_id]

        # 固定自身权重为最高
        if self_max:
            for c_id in range(len(self.p)):
                self.p[c_id][c_id] = np.min(self.p)
            for c_id in range(len(self.p)):
                self.p[c_id][c_id] = np.max(self.p)

    def update_affinity_matrix(self, symmetric=True, self_max=True, normalize=True):
        matrix = np.array(copy.deepcopy(self.p))
        if symmetric:
            matrix = (matrix + matrix.T) * 0.5

        if self_max:
            # 调整自身权重为最大
            for i in range(len(matrix)):
                matrix[i][i] = np.min(matrix)
            for i in range(len(matrix)):
                matrix[i][i] = np.max(matrix)

        if normalize:
            matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

        self.affinity_matrix = matrix
        # print(f"cache keeper: client {self.host.client_id} affinity_matrix {self.affinity_matrix}")
        return matrix

    def update_received_memory(self):
        for c_id in self.host.received_model_dict:
            self.model_memory[c_id] = self.host.received_model_dict[c_id]
            self.topology_weight_memory[c_id] = self.host.received_topology_weight_dict[c_id]
            self.broadcast_weight_memory[c_id] = self.host.received_w_dict[c_id]


    def cache_model(self):
        self.last_local_model = copy.deepcopy(self.host.model)
