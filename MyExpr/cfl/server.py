import copy


class Server(object):

    def __init__(self, args):
        self.client_dic = None
        self.global_model = None
        self.args = args

    def aggregate(self):
        total_sample_num = 0

        # 计算数据总数，汇总local模型
        local_models = {}
        for c_id in self.client_dic:
            total_sample_num += len(self.client_dic[c_id].train_loader)
            model = self.client_dic[c_id].model
            # print("client的model在gpu上:{}".format(self.client_dic[c_id].model))
            if "cuda" in self.args:
                # 这个操作会把model拉出GPU
                model = model.cpu()
            # print("验证client的model还在gpu上:{}".format(self.client_dic[c_id].model))
            params = model.state_dict()
            # 深拷贝，防止影响到原来的模型
            local_models[c_id] = copy.deepcopy(params)

        print("总训练样本数:{}".format(total_sample_num))

        # 借用形状
        global_model = self.client_dic[0].model.cpu().state_dict()

        for k in global_model.keys():
            for i, c_id in enumerate(local_models):
                local_model = local_models[c_id]
                w = len(self.client_dic[c_id].train_loader) / total_sample_num
                if i == 0:
                    global_model[k] = local_model[k] * w
                else:
                    global_model[k] += local_model[k] * w

        self.global_model = global_model
        self.broadcast_global_model()

    def broadcast_global_model(self):
        for c_id in self.client_dic:
            self.client_dic[c_id].model.load_state_dict(self.global_model)
            self.client_dic[c_id].model.to(self.args.device)

