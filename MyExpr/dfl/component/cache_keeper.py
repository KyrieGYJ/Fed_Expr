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
        # self.broadcast_weight_memory = {}
        # 权重缓存（用于计算通信权重，对于本轮没收到的模型复用历史记录）

        # 权重模块
        self.raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))
        self.dif_list = []
        self.broadcast_weight = np.zeros((self.args.client_num_in_total,))
        self.p = np.zeros((self.args.client_num_in_total,))

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
        self.known_set = set()
        self.known_set.add(self.host.client_id)

    # 使用local_train前的本地模型作为基准模型 theta_i(t-1) - theta_n(t)
    def update_model_dif(self, norm=True):
        moniter = True
        base_model = self.last_local_model
        dif_list = np.zeros((self.args.client_num_in_total,))

        for i in range(self.args.client_num_in_total):
            if i == self.host.client_id:
                # 计算本身local_trian后的model_dif
                dif_list[self.host.client_id] = compute_parameter_difference(base_model, self.host.model, norm="l2")
            elif i in self.host.received_model_dict:
                dif_list[i] = compute_parameter_difference(base_model, self.host.received_model_dict[i], norm="l2")
            elif i in self.known_set:
                dif_list[i] = self.dif_list[i]
            else:
                # model_difference的历史缺省值不需要特殊处理，因为它必然大于等于0，几乎不可能等于零
                continue

        self.logger.log_with_name(
            f"[id:{self.host.client_id}]: model_dif[before handle unknown]: {dif_list}",
            self.log_condition & moniter)

        # 处理unknown。
        for i in range(self.args.client_num_in_total):
            if i not in self.known_set:
                # 默认没收到的模型是最差的，这样做会导致一开始没收到的就再也收不到了。
                dif_list[i] = self.dif_list[i]
                # 第一轮赋最差值
                # if self.recorder.rounds == 0:
                #     dif_list[i] = np.max(dif_list)
                # else:
                #     # 后续复用第一轮，这样或许在别的模型下降的时候有机会探索到。
                #     dif_list[i] = self.dif_list[i] # 第一轮的model_dif还是太大了
                    # 后续用下界更新历史，下界更差则不更新
                    # if np.max(dif_list) < self.dif_list[i]:
                    #     dif_list[i] = np.max(dif_list)
                    # else:
                    #     dif_list[i] = self.dif_list[i]

        self.logger.log_with_name(
            f"[id:{self.host.client_id}]: model_dif[after handle unknown]: {dif_list}",
            self.log_condition & moniter)

        # 防止0项干扰norm计算(0被认为是缺省值)
        if np.min(dif_list) == 0.:
            dif_list = np.where(dif_list <= 0., np.partition(dif_list, 1)[1], dif_list)
        if norm:
            dif_list = (dif_list - np.min(dif_list)) / (np.max(dif_list) - np.min(dif_list) + self.epsilon)
        if len(dif_list) != self.args.client_num_in_total:
            print(f"cache_keeper: model dif length error, "
                  f"found {len(dif_list)}, expected {self.args.client_num_in_total}")

        self.logger.log_with_name(
            f"[id:{self.host.client_id}]: model_dif: {dif_list}",
            self.log_condition & moniter)

        self.dif_list = dif_list

    # 更新received_dict包含的，其他则复用历史
    def update_raw_eval_list(self):
        moniter = True
        # 更新已知client集合
        for i in self.host.received_model_dict:
            self.known_set.add(i)

        # 更新raw_loss，缺省值复用历史，历史缺省值默认设置为最大loss
        raw_eval_loss_dict, raw_eval_acc_dict = calc_eval_speed_up_using_cache(self)

        raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))

        for i in range(self.args.client_num_in_total):
            if i in raw_eval_loss_dict:
                raw_eval_loss_list[i] = raw_eval_loss_dict[i]
                raw_eval_acc_list[i] = raw_eval_acc_dict[i]
            else:
                # 处理历史缺省值，防止默认值影响后续特殊处理
                raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
                raw_eval_acc_list[i] = np.max(raw_eval_acc_list)

        self.logger.log_with_name(f"[id:{self.host.client_id}]: raw_eval_loss[before handle unknown] {raw_eval_loss_list}",
                                  self.log_condition & moniter)

        # 历史缺省值，需要特殊处理
        for i in range(self.args.client_num_in_total):
            if i not in self.known_set:
                # 默认没收到的模型是最差的(直接赋值最大的loss和最小acc(默认为效果最差))，这样做会导致一开始没收到的就再也收不到了。
                self.raw_eval_loss_list[i] = np.max(self.raw_eval_loss_list)
                self.raw_eval_acc_list[i] = np.min(self.raw_eval_acc_list)

                # 第一轮赋最差值，后续复用第一轮，这样或许在别的模型下降的时候有机会探索到。
                # if self.recorder.rounds == 0:
                #     raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
                #     raw_eval_acc_list[i] = np.min(raw_eval_acc_list)
                # else:
                #     # 第一轮的最差值还是太大了
                #     raw_eval_loss_list[i] = self.raw_eval_loss_list[i]
                #     raw_eval_acc_list[i] = self.raw_eval_acc_list[i]
                    # if np.max(raw_eval_loss_list) < self.raw_eval_loss_list[i]:
                    #     raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
                    # else:
                    #     raw_eval_loss_list[i] = self.raw_eval_loss_list[i]
                    # if np.min(raw_eval_acc_list) > self.raw_eval_acc_list[i]:
                    #     raw_eval_acc_list[i] = np.min(raw_eval_acc_list)
                    # else:
                    #     raw_eval_acc_list[i] = self.raw_eval_acc_list[i]

        self.raw_eval_loss_list = raw_eval_loss_list
        self.raw_eval_acc_list = raw_eval_acc_list

        self.logger.log_with_name(
            f"[id:{self.host.client_id}]: raw_eval_loss[after handle unknown] {raw_eval_loss_list}",
            self.log_condition & moniter)

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

    def update_p(self):
        # 该策略后续仍然无法解决初始化造成的失联问题
        self.p += self.broadcast_weight
        # 添加指数遗忘
        # r = 0.5
        # self.p = r * self.p + (1 - r) * self.broadcast_weight

    def update_update_weight(self, model_dif_adjust=True):
        moniter = False
        base_loss = self.last_local_loss

        new_update_w_list = base_loss - self.raw_eval_loss_list
        # 不考虑负效果模型(本身除外)
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

    def update_received_memory(self):
        self.logger.log_with_name(f"[id:{self.host.client_id}]: received_list:{self.host.received_model_dict.keys()}",
                                  self.log_condition)
        for c_id in self.host.received_model_dict:
            self.topology_weight_memory[c_id] = self.host.received_topology_weight_dict[c_id]
            # self.broadcast_weight_memory[c_id] = self.host.received_w_dict[c_id]

    def cache_last_local(self):
        self.last_local_model = copy.deepcopy(self.host.model.cpu())
        # 针对广播第一轮，尚未计算eval的情况
        if self.recorder.rounds == 0:
            self.last_local_loss, _ = eval(self.host.model, self.host, self.args)
        else:
            self.last_local_loss = self.raw_eval_loss_list[self.host.client_id]

