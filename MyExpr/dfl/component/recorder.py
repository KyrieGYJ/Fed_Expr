import wandb

# todo 添加基线。
# todo 绘制client_weight曲线。
from MyExpr.dfl.component.history import history


class Recorder(object):

    def __init__(self, client_dict, topology_manager, trainer, broadcaster, topK_selector, data, args):
        # 全局变量
        print("初始化recorder...", end="")
        self.client_dict = client_dict
        self.malignant_dict = None
        self.topology_manager = topology_manager
        self.args = args
        self.data = data

        self.trainer = trainer
        self.broadcaster = broadcaster
        self.topK_selector = topK_selector
        trainer.register_recorder(self)
        broadcaster.register_recorder(self)
        topK_selector.register_recoder(self)
        self.history = history(args)

        # 每个client对应的训练数据下标
        self.train_idx_dict = None

        self.print_log = False
        self.wandb_log = True
        self.rounds = 0
        print("完毕")

    def initialize(self, malignant_dict):
        self.trainer.malignant_dict = malignant_dict
        self.malignant_dict = malignant_dict
        if self.args.trainer_strategy in ["fedavg", "fedprox", "pfedme", "apfl", "fedem"]:
            self.trainer.initialize()

    def record_global_history(self, key, val):
        history_box = self.history.get_history_box(key)
        history_box[self.rounds] = val

    def record_client_history(self, client_id, key, val):
        history_box = self.history.get_history_box(key)
        history_box[client_id][self.rounds] = val

    # # 按照聚类重排client_id(未验证的方法)
    # def reallocate_client_id_by_dist(self):
    #     if self.args.data_distribution == "non-iid_latent":
    #         client_dist_dict = self.data.client_dist_dict
    #         new_client_dic = {}
    #         train_idx_dict = self.data.train_idx_dict
    #         new_train_idx_dict = {}
    #         id = 0
    #         for dist_id in self.data.dist_client_dict:
    #             for c_id in self.data.dist_client_dict[dist_id]:
    #                 client = self.client_dict[c_id]
    #                 client.client_id = id
    #                 new_client_dic[id] = self.client_dict[c_id]
    #                 client_dist_dict[id] = dist_id
    #                 new_train_idx_dict[id] = train_idx_dict[c_id]
    #                 id += 1
    #         self.client_dict = new_client_dic
    #         self.data.train_idx_dict = new_train_idx_dict
    #         print(f"总共{len(self.data.dist_client_dict)}类数据")

