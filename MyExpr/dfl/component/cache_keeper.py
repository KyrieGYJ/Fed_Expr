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
        self.log_condition = self.host.client_id == 35

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
        # self.mutual_update_weight = []  # 采用delta_loss计算
        # self.mutual_update_weight2 = []  # 采用raw_loss计算
        self.mutual_update_weight3 = []  # 采用delta_loss计算，但是加入自因子
        # self.mutual_update_weight4 = []  # 采用raw_acc计算
        # self.mutual_update_weight5 = []  # 采用local_model prior to its current state

        self.sigma = 0.1  # 防止model_dif最大的delta loss丢失
        self.epsilon = 1e-9

        # local_train前的模型缓存
        self.last_local_model = copy.deepcopy(self.host.model.cpu())
        self.last_local_loss = 0.0
        self.known_set = set()
        self.known_set.add(self.host.client_id)

        # 需要尝试的目标，即从未向其发送过的目标 如果考虑新加入的client需要修改
        self.try_set = set()
        for i in range(self.args.client_num_in_total):
            self.try_set.add(i)
        self.try_set.remove(self.host.client_id)

        # self.try_set.add(self.host.client_id)

    # 对于model_dif和eval_Loss，第一轮没收到的client默认赋予平均值，后续则复用这个平均值。
    # 当前没有收到的则复用历史记录，第一轮没有收到的则赋予均值
    # 使用local_train前的本地模型作为基准模型 theta_i(t-1) - theta_n(t)
    def update_model_dif(self, norm=True):
        moniter = False
        base_model = self.last_local_model
        dif_list = np.zeros((self.args.client_num_in_total,))

        avg_dif = 0.0
        for i in range(self.args.client_num_in_total):
            if i == self.host.client_id:
                # 计算本身local_trian后的model_dif
                dif_list[self.host.client_id] = compute_parameter_difference(base_model, self.host.model, norm="l2")
                avg_dif += dif_list[i]
            elif i in self.host.received_model_dict:
                dif_list[i] = compute_parameter_difference(base_model, self.host.received_model_dict[i], norm="l2")
                avg_dif += dif_list[i]
            elif i in self.known_set:
                # todo 可以改成取历史平均值
                dif_list[i] = self.dif_list[i]
                avg_dif += dif_list[i]
            else:
                # 防止影响后续计算
                dif_list[i] = np.max(dif_list)

        self.logger.log_with_name(
            f"[id:{self.host.client_id}]: model_dif[before handle unknown]: {dif_list}",
            self.log_condition & moniter)

        avg_dif = avg_dif / len(self.known_set)

        for i in range(self.args.client_num_in_total):
            if i not in self.known_set:
                # 取平均还是容易攻击
                # dif_list[i] = avg_dif
                # todo 就应该给最差值，后续选到了有遗忘机制，能够拟补前面的差距
                dif_list[i] = np.max(dif_list)

        # # 历史缺省值，需要特殊处理。优先探索try_set，发送过但没有回应的就赋予平均值。
        # for i in range(self.args.client_num_in_total):
        #     if i not in self.try_set:
        #         dif_list[i] = np.min(dif_list)
        #     elif i in self.try_set and i not in self.known_set:
        #         # 取平均还是容易攻击
        #         dif_list[i] = avg_dif

        # # 处理unknown。
        # # todo 优先探索unknown，探索过后没有回应就不再发起探索。
        # for i in range(self.args.client_num_in_total):
        #     if i not in self.known_set:
        #         if i not in self.try_set:
        #             dif_list[i] = np.min(dif_list)
        #         else:
        #             # 取平均还是容易被攻击
        #             dif_list[i] = avg_dif
        #             # 试试取最差
        #             # dif_list[i] = np.max(dif_list)

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
        moniter = False

        # 更新已知client集合，known_set记录接收到过的。 顺便检查
        if len(self.known_set) < self.args.client_num_in_total:
            unknown_malignant = set()
            for i in range(self.args.client_num_in_total - self.args.malignant_num, self.args.client_num_in_total):
                if i not in self.known_set:
                    unknown_malignant.add(i)

            # print(f"client {self.host.client_id} haven't knwon all,"
            #       f" ({self.args.client_num_in_total - len(self.known_set)} left, including {len(unknown_malignant)} malignant)!")

            for i in self.host.received_model_dict:
                self.known_set.add(i)
                # self.try_set.add(i)

        # 更新raw_loss，缺省值复用历史
        raw_eval_loss_dict, raw_eval_acc_dict = calc_eval_speed_up_using_cache(self)

        raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))

        known_loss, known_acc = 0.0, 0.0
        for i in range(self.args.client_num_in_total):
            if i in raw_eval_loss_dict:
                raw_eval_loss_list[i] = raw_eval_loss_dict[i]
                raw_eval_acc_list[i] = raw_eval_acc_dict[i]
                known_loss += raw_eval_loss_list[i]
                known_acc += raw_eval_acc_list[i]
            else:
                # 防止影响后续最值运算
                raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
                raw_eval_acc_list[i] = np.max(raw_eval_acc_list)
        self.logger.log_with_name(f"[id:{self.host.client_id}]: raw_eval_loss[before handle unknown] {raw_eval_loss_list}",
                                  self.log_condition & moniter)

        avg_known_loss, avg_known_acc = known_loss / len(raw_eval_loss_dict), known_acc / len(raw_eval_acc_dict)

        # 初始轮先发送未发送过的，初步试探后再开始正式训练。
        for i in range(self.args.client_num_in_total):
            if i not in self.known_set:
                # 取平均还是容易攻击
                # raw_eval_loss_list[i] = avg_known_loss
                # raw_eval_acc_list[i] = avg_known_acc
                # todo 给最差
                raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
                raw_eval_acc_list[i] = np.min(raw_eval_acc_list)


        # 历史缺省值，需要特殊处理。优先探索try_set，发送过但没有回应的就赋予平均值。
        # 这样有个问题：如果收到了没有尝试发送过的模型，这里会把它当前回合的eval_loss替换成最优，
        # 如果后续再也没有收到了，它的eval_loss就永远是最优。
        # 这是可能发生的，比如当前发送过去的模型在对方的数据集上评估结果很差，对方就不会再发过来
        # for i in range(self.args.client_num_in_total):
        #     if i not in self.try_set:
        #         raw_eval_loss_list[i] = np.min(raw_eval_loss_list)
        #         raw_eval_acc_list[i] = np.max(raw_eval_acc_list)
        #     elif i in self.try_set and i not in self.known_set:
        #         # 取平均还是容易攻击
        #         raw_eval_loss_list[i] = avg_known_loss
        #         raw_eval_acc_list[i] = avg_known_acc

        # # 历史缺省值，需要特殊处理
        # # 优先探索unknown，探索过后没有回应就不再发起探索。
        # for i in range(self.args.client_num_in_total):
        #     if i not in self.known_set:
        #         if i not in self.try_set:
        #             raw_eval_loss_list[i] = np.min(raw_eval_loss_list)
        #             raw_eval_acc_list[i] = np.max(raw_eval_acc_list)
        #         else:
        #             # 取平均还是容易攻击
        #             raw_eval_loss_list[i] = avg_known_loss
        #             raw_eval_acc_list[i] = avg_known_acc
        #             # 试试取最差
        #             # raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
        #             # raw_eval_acc_list[i] = np.min(raw_eval_acc_list)

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

        self.logger.log_with_name(f"[id:{self.host.client_id}]: base_loss {self.last_local_loss}, "
                                  f" raw_eval_loss {self.raw_eval_loss_list}",
                                  self.log_condition & moniter)

    def update_local_eval(self):
        # broadcast之前本地模型进行了local_train，需要重新计算local_eval_loss，与last_local_loss区分开
        local_eval_loss, local_eval_acc = eval(self.host.model, self.host, self.args)
        self.raw_eval_loss_list[self.host.client_id] = local_eval_loss
        self.raw_eval_acc_list[self.host.client_id] = local_eval_acc

    # 本轮广播权重策略。
    # 利用当前的 delta_loss 除以 len(validation_dataset) 做平衡
    # 平衡能够消除client之间数据量差异带来的权值影响。
    def update_broadcast_weight(self, balanced=True):
        moniter = True

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
                                  self.log_condition & moniter)

    # client的全局广播权重向量
    # 采用update_weight加权平均所有broadcast_weight(包括自身)
    # 加权平均相比只采用本身计算的权重的好处在于：能够避免启动带来的陌生client双方长期无法互信的问题。
    def update_p(self):
        # 这种方式还是会有遗漏的部分，而且很容易给恶意模型分到比较好的权重。
        # 添加指数遗忘，防止前面初始值积累出来的差距后续难以追赶
        r = 0.3
        self.p = r * self.p

        # 生成P之前对weight进行归一化，确保每回合影响权重的能力是相同的。 (减去min确保大于等于0)
        aggregated_received_broadcast_weight = (1 - r) * self.mutual_update_weight3[self.host.client_id] \
                                               * ((self.broadcast_weight - np.min(self.broadcast_weight))
                                                  / (np.sum((self.broadcast_weight - np.min(self.broadcast_weight)))
                                                     + self.epsilon))

        for received_id in self.host.received_w_dict:
            aggregated_received_broadcast_weight += (1 - r) * self.mutual_update_weight3[received_id] * \
                                                    ((self.host.received_w_dict[received_id] - np.min(
                                                        self.host.received_w_dict[received_id]))
                                                     / (np.sum((self.host.received_w_dict[received_id] - np.min(
                                                                self.host.received_w_dict[received_id]))
                                                               + self.epsilon)))

        self.p += (1 - r) * aggregated_received_broadcast_weight
        self.logger.log_with_name(f"[id:{self.host.client_id}]: affinity:{self.p}",
                                  self.log_condition)

    def update_update_weight(self, model_dif_adjust=True):
        moniter = False
        base_loss = self.last_local_loss

        # todo raw_loss_list中有填充值，更新时需要做mask，只计算当前接收到的部分
        mask = np.zeros((self.args.client_num_in_total,))
        mask[self.host.client_id] = 1.
        for i in self.host.received_model_dict:
            mask[i] = 1.

        new_update_w_list = base_loss - self.raw_eval_loss_list
        # 不考虑负效果模型(本身除外)
        new_update_w_list = np.where(new_update_w_list >= 0, new_update_w_list, 0)
        new_update_w_list *= mask

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
        self.logger.log_with_name(f"-update_received_memory-[id:{self.host.client_id}]: received_list:{self.host.received_model_dict.keys()}",
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

