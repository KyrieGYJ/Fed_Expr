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

        # 权重模块
        self.raw_eval_loss_list = np.zeros((self.args.client_num_in_total,))
        self.raw_eval_acc_list = np.zeros((self.args.client_num_in_total,))
        self.dif_list = []
        self.broadcast_weight = np.zeros((self.args.client_num_in_total,))
        self.p = np.zeros((self.args.client_num_in_total,))

        self.mutual_update_weight3 = []  # 采用delta_loss计算，但是加入自因子
        # self.mutual_update_weight4 = []  # 采用raw_acc计算
        # self.mutual_update_weight5 = []  # 采用local_model prior to its current state
        self.last_aggregate_broadcast_weight = np.zeros((self.args.client_num_in_total,))

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

    # 对于model_dif和eval_Loss，第一轮没收到的client默认赋予平均值，后续则复用这个平均值。
    # 当前没有收到的则复用历史记录，第一轮没有收到的则赋予均值
    # 使用local_train前的本地模型作为基准模型 theta_i(t-1) - theta_n(t)
    def update_model_dif(self, norm=True):
        moniter = False
        base_model = self.last_local_model
        dif_list = np.zeros((self.args.client_num_in_total,))

        for i in range(self.args.client_num_in_total):
            if i == self.host.client_id:
                # 计算本身local_trian后的model_dif
                dif_list[self.host.client_id] = compute_parameter_difference(base_model, self.host.model, norm="l2")
            elif i in self.host.received_model_dict:
                dif_list[i] = compute_parameter_difference(base_model, self.host.received_model_dict[i], norm="l2")
            elif i in self.known_set:
                # todo futrue work 可以改成取历史平均值
                # 因为每轮只考虑当前接收到模型进行协作更新，所以这个其实没意义。
                dif_list[i] = self.dif_list[i]
            else:
                # 防止影响后续计算
                dif_list[i] = np.max(dif_list)

        self.logger.log_with_name(
            f"[id:{self.host.client_id}]: model_dif[before handle unknown]: {dif_list}",
            self.log_condition & moniter)

        for i in range(self.args.client_num_in_total):
            if i not in self.known_set:
                # 赋予最差值
                dif_list[i] = np.max(dif_list)

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

        for i in range(self.args.client_num_in_total):
            if i in raw_eval_loss_dict:
                raw_eval_loss_list[i] = raw_eval_loss_dict[i]
                raw_eval_acc_list[i] = raw_eval_acc_dict[i]
            else:
                # 防止影响后续最值运算
                raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
                raw_eval_acc_list[i] = np.max(raw_eval_acc_list)
        self.logger.log_with_name(f"[id:{self.host.client_id}]: raw_eval_loss[before handle unknown] {raw_eval_loss_list}",
                                  self.log_condition & moniter)


        # 初始轮先发送未发送过的，初步试探后再开始正式训练。
        for i in range(self.args.client_num_in_total):
            if i not in self.known_set:
                # 不能赋予平均值，如果赋予平均值后一直没有收到对方的消息，则历史一直保持平均值，如果平均值的评价恰好很好，
                # 那么后续会一直误认为其表现很好。
                # 也不能赋予最优值 因为这样有个问题：如果收到了没有尝试发送过的模型，这里会把它当前回合的eval_loss替换成最优，
                # 如果后续再也没有收到了，它的eval_loss就永远是最优。这是可能发生的，比如当前发送过去的模型在对方的数据集上评估结果很差，对方就不会再发过来
                # 赋予最差评分
                raw_eval_loss_list[i] = np.max(raw_eval_loss_list)
                raw_eval_acc_list[i] = np.min(raw_eval_acc_list)

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

        base_loss = self.last_local_loss # todo 这块应该是用当前local train后的模型


        # delta_loss using loss memory
        new_broadcast_w_list = base_loss - self.raw_eval_loss_list

        self.logger.log_with_name(f"[id:{self.host.client_id}]: base_loss:{self.last_local_loss} \n"
                                  f"new_broadcast_w_list:{new_broadcast_w_list}",
                                  self.log_condition & moniter)

        if balanced:
            new_broadcast_w_list /= len(self.host.validation_set)

        # todo 尝试norm(还行)
        new_broadcast_w_list = (new_broadcast_w_list - np.min(new_broadcast_w_list)) \
                               / (np.max(new_broadcast_w_list) - np.min(new_broadcast_w_list) + self.epsilon)
        new_broadcast_w_list = new_broadcast_w_list / max(np.sum(new_broadcast_w_list), self.epsilon)

        # todo 对于正在复用历史(上一轮没收到)的([隐含]或者缺少历史(从未收到过)的部分赋予聚合权重) ，相当于聚合了所有有效节点对某个不信任节点的评分，
        # todo 软性避免有效节点与本身不互信的死锁，同时还避免了盲目探索导致自身特征泄漏。
        # if np.sum(self.last_aggregate_broadcast_weight) != 0:
        #
        #     for i in range(self.args.client_num_in_total):
        #         # todo bug received_model_dict 在回合开始前被清空了，所以相当于每次都直接全部复用了上次的aggregated_received_weight，贯彻始终。
        #         # if i not in self.host.received_model_dict or i not in self.known_set:
        #         if i not in self.host.received_model_dict:
        #             new_broadcast_w_list[i] = self.last_aggregate_broadcast_weight[i]
        #
        #     # 重新归一化 todo 调整比例。
        #     new_broadcast_w_list = new_broadcast_w_list / max(np.sum(new_broadcast_w_list), self.epsilon)

        # 如果有需要复用历史的，融合聚合到的权重进行调整。
        if np.sum(self.last_aggregate_broadcast_weight) != 0 and len(self.host.received_model_dict) != self.args.client_num_in_total:
            r = 0.5 # 融合权重
            if np.sum(self.last_aggregate_broadcast_weight) != 1.0:
                self.last_aggregate_broadcast_weight = self.last_aggregate_broadcast_weight \
                                                       / np.sum(self.last_aggregate_broadcast_weight)

            new_broadcast_w_list = r * new_broadcast_w_list + (1 - r) * self.last_aggregate_broadcast_weight
            # 重新归一化
            new_broadcast_w_list = new_broadcast_w_list / max(np.sum(new_broadcast_w_list), self.epsilon)

        new_broadcast_w_list = (new_broadcast_w_list - np.min(new_broadcast_w_list)) \
                               / (np.max(new_broadcast_w_list) - np.min(new_broadcast_w_list) + self.epsilon)
        new_broadcast_w_list = new_broadcast_w_list / max(np.sum(new_broadcast_w_list), self.epsilon)

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
        # 因为mutual_update_weight3也进行过归一化，所以aggregated_received_broadcast_weight意识归一化的
        shift_positive_weight = self.broadcast_weight - np.min(self.broadcast_weight)
        normed_broadcast_weight = shift_positive_weight / (np.sum(shift_positive_weight) + self.epsilon)
        aggregated_received_broadcast_weight = self.mutual_update_weight3[self.host.client_id] * normed_broadcast_weight

        for received_id in self.host.received_w_dict:
            shift_positive_weight = self.host.received_w_dict[received_id] - np.min(
                self.host.received_w_dict[received_id])
            normed_broadcast_weight = shift_positive_weight / (np.sum(shift_positive_weight) + self.epsilon)
            aggregated_received_broadcast_weight += self.mutual_update_weight3[received_id] * normed_broadcast_weight

        self.p += (1 - r) * aggregated_received_broadcast_weight
        self.logger.log_with_name(f"[id:{self.host.client_id}]: affinity:{self.p}",
                                  self.log_condition)

        # todo future-work 尝试采用聚合的广播权重去更新本地广播权重 (不能直接聚合，要先剔除恶意值)
        self.last_aggregate_broadcast_weight = aggregated_received_broadcast_weight
        self.logger.log_with_name(f"[id:{self.host.client_id}]: aggregated_received_broadcast_weight:{aggregated_received_broadcast_weight}",
                                  self.log_condition)


    def update_update_weight(self, model_dif_adjust=True):
        moniter = False
        base_loss = self.last_local_loss

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
            # new_update_w_list = new_update_w_list * (1 - self.dif_list + self.sigma)  # 和模型差距成反比
            # todo 试试除法的做法 (好像不错)
            new_update_w_list = new_update_w_list / (self.dif_list + self.sigma) # 和模型差距成反比

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

    def cache_last_local(self):
        self.last_local_model = copy.deepcopy(self.host.model.cpu())
        # 针对广播第一轮，尚未计算eval的情况
        if self.recorder.rounds == 0:
            self.last_local_loss, _ = eval(self.host.model, self.host, self.args)
        else:
            self.last_local_loss = self.raw_eval_loss_list[self.host.client_id]

