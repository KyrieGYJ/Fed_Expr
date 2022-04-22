import copy
import numpy as np
from MyExpr.utils import calc_eval, compute_parameter_difference, eval, calc_eval_speed_up_using_cache
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
        self.topology_weight_memory = {}
        self.broadcast_weight_memory = {}
        # 权重缓存（用于计算通信权重，对于本轮没收到的模型复用历史记录）

        # 权重模块
        self.raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))
        self.delta_loss_list = []
        self.dif_list = []
        self.broadcast_weight = np.zeros((self.args.client_num_in_total,))

        # todo 删除多余逻辑
        self.mutual_update_weight = []  # 采用delta_loss计算
        self.mutual_update_weight2 = []  # 采用raw_loss计算
        self.mutual_update_weight3 = []  # 采用delta_loss计算，但是加入自因子
        self.mutual_update_weight4 = []  # 采用raw_acc计算
        self.mutual_update_weight5 = []  # 采用local_model prior to its current state

        self.sigma = 0.1  # 防止model_dif最大的delta loss丢失
        self.epsilon = 1e-9

        # local_train前的模型缓存
        self.last_local_model = copy.deepcopy(self.host.model.cpu())
        self.last_local_loss = 0.0

        self.p = None # 原始权重矩阵
        self.affinity_matrix = None # 调整后的权重矩阵（p和affinity应该是等价的才行）
        self.known_set = set()

        # self.update_uw = None

    def initialize(self):
        self.p = np.zeros([self.args.client_num_in_total, self.args.client_num_in_total],
                         dtype=np.float64)
        self.update_affinity_matrix()


    # 使用local_train前的本地模型作为基准模型 theta_i(t-1) - theta_n(t)
    def update_model_dif(self, norm=True):
        base_model = self.last_local_model
        dif_list = np.zeros((self.args.client_num_in_total,))

        for i in range(self.args.client_num_in_total):
            if i == self.host.client_id:
                # 计算本身local_trian后的model_dif
                dif_list[self.host.client_id] = compute_parameter_difference(base_model, self.host.model, norm="l2")
            elif i in self.host.received_model_dict:
                dif_list[i] = compute_parameter_difference(base_model, self.host.received_model_dict[i], norm="l2")
            # 原未考虑历史值也会缺省
            else:
                dif_list[i] = self.dif_list[i]
            # elif i in self.known_set:
            #     dif_list[i] = self.dif_list[i]
            # else:
            #     # model_difference的历史缺省值不需要特殊处理，因为它必然大于等于0，几乎不可能等于零
            #     continue

        # 处理 unknown，默认没收到的模型是最差的。
        # for i in range(self.args.client_num_in_total):
        #     if i not in self.known_set:
        #         dif_list[i] = np.max(dif_list)

        # 防止0项干扰norm计算(0被认为是缺省值)
        if np.min(dif_list) == 0.:
            dif_list = np.where(dif_list <= 0., np.partition(dif_list, 1)[1], dif_list)
        if norm:
            dif_list = (dif_list - np.min(dif_list)) / (np.max(dif_list) - np.min(dif_list) + self.epsilon)
        if len(self.delta_loss_list) != self.args.client_num_in_total:
            print(f"cache_keeper: model dif length error, "
                  f"found {len(dif_list)}, expected {self.args.client_num_in_total}")

        self.dif_list = dif_list

    # 更新received_dict包含的，其他则复用历史
    def update_raw_eval_list(self):
        # 更新已知client集合
        # for i in self.host.received_model_dict:
        #     self.known_set.add(i)

        # 更新raw_loss，缺省值复用历史，历史缺省值默认设置为最大loss
        raw_eval_loss_dict, raw_eval_acc_dict = calc_eval_speed_up_using_cache(self)
        self.delta_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))

        for i in range(self.args.client_num_in_total):
            self.raw_eval_loss_list[i] = raw_eval_loss_dict[i]
            self.raw_eval_acc_list[i] = raw_eval_acc_dict[i]

        # for i in range(self.args.client_num_in_total):
        #     if i in raw_eval_loss_dict:
        #         self.raw_eval_loss_list[i] = raw_eval_loss_dict[i]
        #         self.raw_eval_acc_list[i] = raw_eval_acc_dict[i]
        #     else:
        #         # 处理历史缺省值，防止默认值影响后续特殊处理
        #         self.raw_eval_loss_list[i] = np.max(self.raw_eval_loss_list)
        #         self.raw_eval_acc_list[i] = np.max(self.raw_eval_acc_list)

        # 历史缺省值，需要特殊处理，直接赋值最大的loss(默认为效果最差)
        for i in range(self.args.client_num_in_total):
            if i not in self.known_set:
                self.raw_eval_loss_list[i] = np.max(self.raw_eval_loss_list)
                self.raw_eval_acc_list[i] = np.max(self.raw_eval_acc_list)

        # check
        if len(self.raw_eval_loss_list) != self.args.client_num_in_total:
            self.logger.log_with_name(f"[id:{self.host.client_id}]: delta_loss length error"
                                      f", found {len(self.raw_eval_loss_list)}"
                                      f", expected {self.args.client_num_in_total}", True)

        self.logger.log_with_name(f"[id:{self.host.client_id}]: raw_eval_loss {self.raw_eval_loss_list}",
                                  self.log_condition)

    def update_local_eval(self):
        # broadcast之前本地模型进行了local_train，需要重新计算local_eval_loss，与last_local_loss区分开
        local_eval_loss, local_eval_acc = eval(self.host.model, self.host, self.args)
        self.raw_eval_loss_list[self.host.client_id] = local_eval_loss
        self.raw_eval_acc_list[self.host.client_id] = local_eval_acc

    def update_broadcast_weight(self, balanced=True):
        moniter = False
        base_loss = self.last_local_loss
        # delta_loss using loss memory
        new_broadcast_w_list = base_loss - self.raw_eval_loss_list

        self.logger.log_with_name(f"[id:{self.host.client_id}]: base_loss:{self.last_local_loss} \n"
                                  f"new_broadcast_w_list:{new_broadcast_w_list}",
                                  self.log_condition & moniter)

        if balanced:
            new_broadcast_w_list /= len(self.host.validation_set)

        self.broadcast_weight = new_broadcast_w_list
        self.logger.log_with_name(f"[id:{self.host.client_id}]: balanced_broadcast_w_list:{new_broadcast_w_list}",
                                  self.log_condition)

    def update_update_weight(self, model_dif_adjust=True):
        moniter = False
        base_loss = self.last_local_loss

        new_update_w_list = base_loss - self.raw_eval_loss_list
        new_update_w_list = np.where(new_update_w_list >= 0, new_update_w_list, 0)

        self.logger.log_with_name(f"[id:{self.host.client_id}]: base_loss:{base_loss} \n"
                                  f"raw_update_w_list:{new_update_w_list}",
                                  self.log_condition & moniter)

        if model_dif_adjust:
            # 接收到了新模型，计算新model_dif，对于没接收到的复用历史数据
            self.update_model_dif()
            new_update_w_list = new_update_w_list * (1 - self.dif_list + self.sigma)  # 和模型差距成反比

        self.logger.log_with_name(f"[id:{self.host.client_id}]: update_w_list[model_dif_adjust]:{new_update_w_list}",
                                  self.log_condition & moniter)

        # norm
        new_update_w_list /= max(np.sum(new_update_w_list), self.epsilon)

        self.logger.log_with_name(f"[id:{self.host.client_id}]: update_w_list[norm]:{new_update_w_list}",
                                  self.log_condition & moniter)

        # 考虑给self一个保底权重
        threshold = self.args.aggregate_threshold
        if new_update_w_list[self.host.client_id] < threshold:
            new_update_w_list[self.host.client_id] = (threshold * (1 - new_update_w_list[self.host.client_id])) \
                                                     / (1 - threshold)
            new_update_w_list /= max(np.sum(new_update_w_list), self.epsilon)

        self.mutual_update_weight3 = new_update_w_list
        self.logger.log_with_name(f"[id:{self.host.client_id}]: update_w_list[threshold]:{new_update_w_list}",
                                  self.log_condition)

    # todo 整合成一个方法
    def update_affinity_map(self):
        self.update_p()
        self.update_affinity_matrix()

    def update_p(self, self_max=True):
        self.p[self.host.client_id] += self.broadcast_weight

        # todo 个人权重矩阵应该要取消掉
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
            matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + self.epsilon)

        self.affinity_matrix = matrix
        # print(f"cache keeper: client {self.host.client_id} affinity_matrix {self.affinity_matrix}")
        return matrix

    def update_received_memory(self):
        for c_id in self.host.received_model_dict:
            self.topology_weight_memory[c_id] = self.host.received_topology_weight_dict[c_id]
            self.broadcast_weight_memory[c_id] = self.host.received_w_dict[c_id]
            # todo 历史缺省值不好处理，所以可能就不看client的权重矩阵了，只保存向量，然后主要看通信频率图


    def cache_last_local(self):
        self.last_local_model = copy.deepcopy(self.host.model.cpu())
        # 针对广播第一轮，尚未计算eval的情况
        if self.host.client_id >= len(self.raw_eval_loss_list):
            self.last_local_loss, _ = eval(self.host.model, self.host, self.args)
        else:
            self.last_local_loss = self.raw_eval_loss_list[self.host.client_id]

