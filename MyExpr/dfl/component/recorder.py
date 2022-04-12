import wandb


class Recorder(object):

    def __init__(self, client_dic, topology_manager, trainer, broadcaster, topK_selector, data, args):
        # 全局变量
        print("初始化recorder...", end="")
        self.client_dic = client_dic
        self.topology_manager = topology_manager
        self.args = args
        self.data = data

        self.trainer = trainer
        self.broadcaster = broadcaster
        self.topK_selector = topK_selector
        trainer.register_recorder(self)
        broadcaster.register_recorder(self)
        topK_selector.register_recoder(self)

        trainer.non_iid_test_set = data.non_iid_test_set
        trainer.broadcaster = broadcaster
        trainer.test_non_iid = data.test_non_iid
        trainer.client_class_dic = data.client_class_dic
        trainer.class_client_dic = data.class_client_dic
        trainer.dist_client_dict = data.dist_client_dict
        trainer.client_dist_dict = data.client_dist_dict
        trainer.test_non_iid = data.test_non_iid
        trainer.test_loader = data.test_all
        trainer.test_data = data.test_data

        # 每个client对应的训练数据下标
        self.train_idx_dict = None

        self.print_log = False
        self.wandb_log = True
        self.rounds = 0
        print("完毕")

    # 按照聚类重排client_id(未验证的方法)
    def reallocate_client_id_by_dist(self):
        if self.args.data_distribution == "non-iid_latent":
            client_dist_dict = self.data.client_dist_dict
            new_client_dic = {}
            train_idx_dict = self.data.train_idx_dict
            new_train_idx_dict = {}
            id = 0
            for dist_id in self.data.dist_client_dict:
                for c_id in self.data.dist_client_dict[dist_id]:
                    client = self.client_dic[c_id]
                    client.client_id = id
                    new_client_dic[id] = self.client_dic[c_id]
                    client_dist_dict[id] = dist_id
                    new_train_idx_dict[id] = train_idx_dict[c_id]
                    id += 1
            self.client_dic = new_client_dic
            self.data.train_idx_dict = new_train_idx_dict
            print(f"总共{len(self.data.dist_client_dict)}类数据")
        # for c_id in range(client_num_in_total):
        #     print(new_client_dist_dict[c_id])

    def next_epoch(self):
        self.epoch += 1
