import copy
import numpy as np
from MyExpr.utils import calc_eval, compute_parameter_difference, eval
from MyExpr.dfl.component.logger import logger


class keeper(object):
    """
    主要是解耦权重模块，不然client太重了
    """

    def __init__(self, host):
        self.host = host
        self.recorder = host.recorder
        self.args = self.recorder.args
        self.logger = logger(self, "cache_keeper")
        self.log_condition = self.host.client_id == 0


        # 接收缓存
        self.model_memory = {}
        self.topology_weight_memory = {}
        self.broadcast_weight_memory = {}
        # 权重缓存（用于计算通信权重，对于本轮没收到的模型复用历史记录）
        self.delta_loss_memory = {}
        self.model_dif_memory = {}

        # 权重模块 todo 改成list
        # self.delta_loss_dict = {}
        # self.delta_loss_prior_dict = {}
        # self.raw_eval_loss_dict = {}
        # self.raw_eval_acc_dict = {}
        self.raw_eval_loss_list = []
        self.raw_eval_acc_list = []
        self.delta_loss_list = []
        self.dif_list = []
        self.dif_list_prior = []  # 采用local_model prior to its current state
        self.broadcast_weight = []

        self.mutual_update_weight = []  # 采用delta_loss计算
        self.mutual_update_weight2 = []  # 采用raw_loss计算
        self.mutual_update_weight3 = []  # 采用delta_loss计算，但是加入自因子
        self.mutual_update_weight4 = []  # 采用raw_acc计算
        self.mutual_update_weight5 = []  # 采用local_model prior to its current state
        self.sigma = 0.1  # 防止model_dif最大的delta loss丢失

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
        # self.dif_list_prior = self.update_model_difference(local_model=self.last_local_model)

        self.update_uw()
        self.update_bw()

    def update_affinity_map(self):
        self.update_p()
        self.update_affinity_matrix()

    # 使用local_train后的本地模型作为基准模型 delta_loss = L_i(t) - L_n(t)
    def update_delta_loss(self):
        # raw_eval_loss_dict包含了self
        raw_eval_loss_dict, raw_eval_acc_dict = calc_eval(self.host, self.model_memory, self.args)
        local_loss = raw_eval_loss_dict[self.host.client_id]
        # [0. for _ in range(self.args.client_num_in_total)]
        self.delta_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))
        for i in range(self.args.client_num_in_total):
            self.delta_loss_list[i] = local_loss - raw_eval_loss_dict[i]
            self.raw_eval_loss_list[i] = raw_eval_loss_dict[i]
            self.raw_eval_acc_list[i] = raw_eval_acc_dict[i]
        # self.delta_loss_list = np.array(self.delta_loss_list)

        # check
        if len(self.delta_loss_list) != self.args.client_num_in_total:
            self.logger.log_with_name(f"[id:{self.host.client_id}]: delta_loss length error"
                                      f", found {len(self.delta_loss_list)}"
                                      f", expected {self.args.client_num_in_total}", True)

        self.logger.log_with_name(f"[id:{self.host.client_id}]: delta_loss {self.delta_loss_list}", self.log_condition)

    # todo last_local_model好像有点问题
    def update_delta_loss_using_prior(self):
        raw_eval_loss_dict, raw_eval_acc_dict = calc_eval(self.host, self.model_memory, self.args)
        local_loss = eval(self.last_local_model, self.host, self.args)
        self.delta_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))
        for i in range(self.args.client_num_in_total):
            self.delta_loss_list[i] = local_loss - raw_eval_loss_dict[i]
            self.raw_eval_loss_list[i] = raw_eval_loss_dict[i]
            self.raw_eval_acc_list[i] = raw_eval_acc_dict[i]
        self.logger.log_with_name(f"[id:{self.host.client_id}]: delta_loss {self.delta_loss_list}", self.log_condition)
        # 使用local_train前的本地模型作为基准模型 delta_loss_prior = L_i(t-1) - L_n(t)
        # local_loss_prior, _ = eval(self.last_local_model, self.host, self.args)
        # self.delta_loss_prior_dict = {}
        # for c_id in self.raw_eval_loss_dict:
        #     self.delta_loss_prior_dict[c_id] = local_loss_prior - self.raw_eval_loss_dict[c_id]
        # print(f"cache keeper: client {self.host.client_id} keys-delta_loss_dict:{self.delta_loss_dict.keys()} \n "
        #       f"keys-delta_loss_prior_dict:{self.delta_loss_prior_dict.keys()}")
        pass

    def update_model_difference(self, base_model=None, norm=True):
        epsilon = 1e-9
        # mode_name = "local model" if local_model is None else "local prior model"
        # local_model = self.host.model if local_model is None else local_model
        base_model = self.host.model
        dif_list = np.zeros((self.args.client_num_in_total,))
        for i in range(self.args.client_num_in_total):
            # todo 确认model_memory后回来改, prior同
            if i == self.host.client_id:
                continue
            dif_list[i] = compute_parameter_difference(base_model, self.model_memory[i], norm="l2")
        # 添加自身
        dif_list[self.host.client_id] = np.partition(dif_list, 1)[1]
        # dif_list[self.host.client_id] = compute_parameter_difference(base_model, self.host.model, norm="l2")
        # # 防止0项干扰norm计算
        # if np.min(dif_list) == 0.:
        #     dif_list = np.where(dif_list <= 0., np.partition(dif_list, 1)[1], dif_list)

        # print(f"cache_keeper: client {self.host.client_id} raw model_dif:{dif_list}, max:{np.max(dif_list)}, min:{np.min(dif_list)}")
        if norm:
            dif_list = (dif_list - np.min(dif_list)) / (np.max(dif_list) - np.min(dif_list) + epsilon)  # norm

        if len(self.delta_loss_list) != self.args.client_num_in_total:
            print(f"cache_keeper: delta_loss length error, "
                  f"found {len(dif_list)}, expected {self.args.client_num_in_total}")

        self.logger.log_with_name(f"[id:{self.host.client_id}]: model_dif {self.delta_loss_list}", self.log_condition)
        return dif_list

    def update_model_difference_using_prior(self, norm=True):
        epsilon = 1e-9
        base_model = self.last_local_model
        dif_list = np.zeros((self.args.client_num_in_total,))
        for i in range(self.args.client_num_in_total):
            if i == self.host.client_id:
                continue
            dif_list[i] = compute_parameter_difference(base_model, self.model_memory[i], norm="l2")
        # 添加自身
        dif_list[self.host.client_id] = compute_parameter_difference(base_model, self.host.model, norm="l2")
        # 防止0项干扰norm计算
        if np.min(dif_list) == 0.:
            dif_list = np.where(dif_list <= 0., np.partition(dif_list, 1)[1], dif_list)
        # dif_list = np.array(dif_list)
        # print(f"cache_keeper[id:{self.host.client_id}]: raw model dif {dif_list}")

        if len(self.delta_loss_list) != self.args.client_num_in_total:
            print(f"cache_keeper: delta_loss length error, "
                  f"found {len(dif_list)}, expected {self.args.client_num_in_total}")
        return dif_list


    def check_update_weight(self, model_dif_adjust=True):
        self.update_uw1(model_dif_adjust)
        self.update_uw2(model_dif_adjust)
        self.update_uw3(model_dif_adjust)
        self.update_uw4(model_dif_adjust)
        self.update_uw5(model_dif_adjust)

    def update_uw1(self, model_dif_adjust=True):
        pass
        # delta_loss = self.delta_loss_dict
        # new_update_w_list = []
        #
        # epsilon = 1e-9
        # for i in range(self.args.client_num_in_total):
        #     # relu
        #     if i in delta_loss and delta_loss[i] >= 0:
        #         new_update_w_list.append(copy.deepcopy(delta_loss[i]))
        #     else:
        #         new_update_w_list.append(0)
        #
        # if model_dif_adjust:
        #     new_update_w_list *= (1 - self.dif_list + self.sigma) # 和模型差距成反比
        #
        # # 更新权重需要norm，邻居模型贡献的总权重为1
        # norm_factor = max(np.sum(new_update_w_list), epsilon)
        # new_update_w_list = np.array(new_update_w_list) / norm_factor
        #
        # # print(f"client {self.host.client_id} new_update_w_list : {new_update_w_list}")
        # self.mutual_update_weight = new_update_w_list
        # print(f"cache keeper: client{self.host.client_id} new_update_w_list {new_update_w_list}")

    def update_uw2(self, model_dif_adjust=True):
        pass
        # raw_eval_loss = self.raw_eval_loss_dict
        # new_update_w_list = []
        #
        # epsilon = 1e-9
        # for i in range(self.args.client_num_in_total):
        #     # relu
        #     if i in raw_eval_loss and self.delta_loss_dict[i] >= 0:
        #         new_update_w_list.append(1 / (copy.deepcopy(raw_eval_loss[i]) + epsilon))
        #     else:
        #         new_update_w_list.append(0)
        #
        # if model_dif_adjust:
        #     new_update_w_list *= (1 - self.dif_list + self.sigma)  # 和模型差距成反比
        #
        # # 更新权重需要norm，邻居模型贡献的总权重为1
        # norm_factor = max(np.sum(new_update_w_list), epsilon)
        # new_update_w_list = np.array(new_update_w_list) / norm_factor
        #
        # # print(f"client {self.host.client_id} new_update_w_list : {new_update_w_list}")
        # self.mutual_update_weight2 = new_update_w_list
        # print(f"cache keeper: client{self.host.client_id} new_update_w2_list {new_update_w_list}")

    def update_uw3(self, model_dif_adjust=True):
        # new_update_w_list = np.array(new_update_w_list)
        # delta_loss = self.delta_loss_dict
        new_update_w_list = copy.deepcopy(self.delta_loss_list)
        # print(f"cache_keeper[id:{self.host.client_id}]: [raw] new_update_w_list:{new_update_w_list}")
        new_update_w_list = np.where(new_update_w_list >=0, new_update_w_list+self.sigma, 0)
        # print(f"cache_keeper[id:{self.host.client_id}]: [var_relu] new_update_w_list:{new_update_w_list}")

        epsilon = 1e-9

        if model_dif_adjust:
            # delta_loss + sigma: 防止因为delta_loss为0而在聚合时忽略自身
            # 1 - model_dif + sigma: 防止model_dif最大者被忽略
            new_update_w_list = new_update_w_list * (1 - self.dif_list + self.sigma)  # 和模型差距成反比
        # print(f"cache_keeper[id:{self.host.client_id}]: [model_dif_adjust] new_update_w_list:{new_update_w_list}")

        # 更新权重需要norm，邻居模型贡献的总权重为1
        norm_factor = max(np.sum(new_update_w_list), epsilon)
        new_update_w_list /= norm_factor

        self.logger.log_with_name(f"[id:{self.host.client_id}]: new_update_w_list:{new_update_w_list }", self.log_condition)
        self.mutual_update_weight3 = new_update_w_list

    def update_uw4(self, model_dif_adjust=True):
        pass
        # raw_acc = self.raw_eval_acc_dict
        # new_update_w_list = []
        #
        # epsilon = 1e-9
        # for i in range(self.args.client_num_in_total):
        #     # relu 忽略负效果模型
        #     if i in raw_acc and self.delta_loss_dict[i] >= 0:
        #         new_update_w_list.append(copy.deepcopy(raw_acc[i]))
        #     else:
        #         new_update_w_list.append(0)
        # new_update_w_list = np.array(new_update_w_list)
        #
        # if model_dif_adjust:
        #     new_update_w_list = new_update_w_list * (1 - self.dif_list + self.sigma)  # 和模型差距成反比
        #
        # # 更新权重需要norm，邻居模型贡献的总权重为1
        # norm_factor = max(np.sum(new_update_w_list), epsilon)
        # new_update_w_list = np.array(new_update_w_list) / norm_factor
        # self.mutual_update_weight4 = new_update_w_list
        # print(f"cache keeper: client{self.host.client_id} new_update_w4_list {new_update_w_list}")

    def update_uw5(self, model_dif_adjust=True):
        pass
        # delta_loss = self.delta_loss_prior_dict
        # new_update_w_list = []
        #
        # epsilon = 1e-9
        # for i in range(self.args.client_num_in_total):
        #     # relu 忽略负效果模型
        #     if i in delta_loss and delta_loss[i] >= 0:
        #         new_update_w_list.append(copy.deepcopy(delta_loss[i]))
        #     else:
        #         new_update_w_list.append(0)
        # new_update_w_list = np.array(new_update_w_list)
        # # print(f"cache keeper: before adjustment client{self.host.client_id} delta_loss - max:{np.max(new_update_w_list)}, true min:{np.partition(new_update_w_list, 2)[2]}, sum: {np.sum(new_update_w_list)}")
        # if model_dif_adjust:
        #     # 1 - model_dif + sigma: 防止model_dif最大者被忽略
        #     new_update_w_list = new_update_w_list * (1 - self.dif_list_prior + self.sigma)  # 和模型差距成反比

        # todo 改完用 c60_ours02 、c60_ours05对比


        # 更新权重需要norm，邻居模型贡献的总权重为1
        norm_factor = max(np.sum(new_update_w_list), epsilon)
        new_update_w_list = np.array(new_update_w_list) / norm_factor
        self.mutual_update_weight5 = new_update_w_list
        # print(f"cache keeper: client{self.host.client_id} new_update_w5_list {new_update_w_list}")
        # print(f"cache keeper: after adjustment client{self.host.client_id} new_update_w5_list - max:{np.max(new_update_w_list)}, true min:{np.partition(new_update_w_list, -2)[-2]}, sum: {np.sum(new_update_w_list)}")

    def update_bw(self, balanced=False, model_dif_adjust=True):
        new_broadcast_w_list = copy.deepcopy(self.delta_loss_list)

        # 用raw_loss效果并不那么好
        # raw_loss = self.raw_eval_loss_dict
        # new_broadcast_w_list = []
        #
        # for i in range(self.args.client_num_in_total):
        #     if i in raw_loss:
        #         new_broadcast_w_list.append(copy.deepcopy(raw_loss[i]))
        #     else:
        #         new_broadcast_w_list.append(0)

        # print(f"before model_dif_adjust client {self.host.client_id} new_broadcast_w_list : {new_broadcast_w_list}")

        if model_dif_adjust:
            # new_broadcast_w_list *= (1 - self.dif_list + self.sigma) # 和模型差距成反比
            # delta_loss小于零的时候，直接乘上model_dif + sigma
            new_broadcast_w_list = np.where(new_broadcast_w_list >= 0,
                                            new_broadcast_w_list * (1 - self.dif_list + self.sigma),
                                            new_broadcast_w_list * (self.dif_list + self.sigma))

        # print(f"model_dif: {self.dif_list}")
        # print(f"1 - model_dif: {(1 - self.dif_list + self.sigma)}")
        # new_broadcast_w_list = np.array(new_broadcast_w_list)
        # print(f"after model_dif_adjust client {self.host.client_id} new_broadcast_w_list : {new_broadcast_w_list}")
        if balanced:
            new_broadcast_w_list /= len(self.host.validation_set)

        self.logger.log_with_name(f"[id:{self.host.client_id}]: new_broadcast_w_list:{new_broadcast_w_list}", self.log_condition)
        self.broadcast_weight = new_broadcast_w_list
        # print(f"cache keeper: client{self.host.client_id} new_broadcast_w_list {new_broadcast_w_list}")

    def update_p(self, self_max=True):
        self.p[self.host.client_id] += self.broadcast_weight

        for neighbor_id in self.broadcast_weight_memory:
            self.p[neighbor_id] += self.broadcast_weight_memory[neighbor_id]

        # 固定自身权重为最高
        if self_max:
            row, col = np.diag_indices_from(self.p)
            self.p[row, col] = np.min(self.p)
            self.p[row, col] = np.max(self.p)

    def update_affinity_matrix(self, symmetric=True, self_max=True, normalize=True):
        matrix = np.array(copy.deepcopy(self.p))
        if symmetric:
            matrix = (matrix + matrix.T) * 0.5

        if self_max:
            row, col = np.diag_indices_from(matrix)
            matrix[row, col] = np.min(matrix)
            matrix[row, col] = np.max(matrix)

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
        self.model_memory[self.host.client_id] = self.host.model # 这是当前的model


    def cache_model(self):
        self.last_local_model = self.host.model

# import time

# if __name__ == '__main__':
#     a = np.array([1, 3, 10, 2, 5, 3])
# #     # 把第2大放到第2个
#     second_min = np.partition(a, 2)[2]
#     print(np.partition(a, -2)[-2], np.partition(a, 0)[0], np.partition(a, 1)[1], np.partition(a, 2)[2])
#     print(np.where(a<=0., second_min, a))
    # print(np.where(a < second_min, a+999, a))
#     # b = np.array([[80, 89, 86, 67, 79],
#     #                 [78, 97, 89, 67, 81],
#     #                 [90, 94, 78, 67, 74],
#     #                 [91, 91, 90, 67, 69],
#     #                 [76, 87, 75, 67, 86]])
#     start = time.time()
#     b = np.random.randint(0, 100000, size=(100, 100))
#     row, col = np.diag_indices_from(b)
#     b[row, col] = np.min(b)
#     b[row, col] = np.max(b)
#     end = time.time()
#     print(f"spend {end - start} s")
#     start = time.time()
#     b = np.random.randint(0, 100000, size=(100, 100))
#     for i in range(100):
#         b[i][i] = np.min(b)
#     for i in range(100):
#         b[i][i] = np.max(b)
#     end = time.time()
#     print(f"spend {end - start} s")

