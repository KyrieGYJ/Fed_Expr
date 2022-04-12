import copy
import numpy as np
import time


class Server(object):

    def __init__(self, args):
        self.client_dic = None
        self.global_model = None
        self.args = args

    def aggregate(self):
        total_sample_num = 0

        # 取随机
        K = min(int(self.args.broadcast_K * self.args.client_num_in_total), self.args.client_num_in_total)
        np.random.seed(int(round(time.time())))  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(self.args.client_num_in_total), K, replace=False)

        # 计算数据总数，汇总local模型
        local_models = {}
        for c_id in client_indexes:
            total_sample_num += len(self.client_dic[c_id].train_set)
            # print(f"client {c_id} train data length {len(self.client_dic[c_id].train_set)}")
            model = self.client_dic[c_id].model
            if "cuda" in self.args:
                # 这个操作会把model拉出GPU
                model = model.cpu()
            params = model.state_dict()
            # 深拷贝，防止影响到原来的模型
            local_models[c_id] = copy.deepcopy(params)

        # print("总训练样本数:{}".format(total_sample_num))

        # 借用形状
        global_model = self.client_dic[0].model.cpu().state_dict()

        for k in global_model.keys():
            for i, c_id in enumerate(local_models):
                local_model = local_models[c_id]
                w = len(self.client_dic[c_id].train_set) / total_sample_num
                if i == 0:
                    global_model[k] = local_model[k] * w
                else:
                    global_model[k] += local_model[k] * w

        self.global_model = global_model
        self.broadcast_global_model(client_indexes)

    def broadcast_global_model(self, client_indexes):
        for c_id in client_indexes:
            self.client_dic[c_id].model.load_state_dict(copy.deepcopy(self.global_model))
            # 将模型恢复到原来的GPU上
            self.client_dic[c_id].model.to(self.args.device)
