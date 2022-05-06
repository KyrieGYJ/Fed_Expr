import wandb
from tqdm import tqdm


# decentralized training simulator
class Ours_Trainer(object):

    def __init__(self, args):
        self.args = args
        self.recorder = None
        self.client_dict = None
        self.malignant_dict = None
        self.broadcaster = None

        self.best_accuracy = 0.
        # 训练策略
        self.train = None
        self.strategy = None
        self.use(args.trainer_strategy)

    def register_recorder(self, recorder):
        self.recorder = recorder
        self.client_dict = recorder.client_dict
        self.broadcaster = recorder.broadcaster

    def use(self, strategy):
        print(f"Trainer use strategy:{strategy}")
        self.strategy = strategy
        self.train = self.weighted_model_interpolation

    def weighted_model_interpolation(self):
        self.cache_model() # 记录local train之前的model
        # before local train
        self.local() # 本地训练
        # before broadcast
        self.update_broadcast_weight() # 更新广播权重 (用到上回接受到的模型loss以及上回接受到的客户下标信息)
        self.clear_received()  # 广播前清空received_list
        self.broadcast() # 广播 (需要先清空received_list)
        # before update
        self.cache_received() # 缓存当前的接受信息到cache_keeper(已经废弃)
        self.update_update_weight() # 更新插值权重
        # update
        self.weighted_interpolation_update() # 权重插值生成新模型

    # 广播
    def broadcast(self):
        for c_id in tqdm(self.client_dict, desc="benign broadcast"):
            self.client_dict[c_id].broadcast()
        for c_id in tqdm(self.malignant_dict, desc="malignant broadcast"):
            self.malignant_dict[c_id].broadcast()

    # 更新广播权重
    def update_broadcast_weight(self):
        for sender_id in tqdm(self.client_dict, desc="update broadcast weight"):
            if self.recorder.rounds > 0:
                # broadcast前更新了本地模型，更新local_eval
                self.client_dict[sender_id].cache_keeper.update_local_eval()
                self.client_dict[sender_id].cache_keeper.update_broadcast_weight()
                self.client_dict[sender_id].cache_keeper.update_p()
            else:
                # 第一轮还没收到模型，raw_eval_loss为空，无法更新broadcast_weight，更没法更新p。
                self.client_dict[sender_id].cache_keeper.update_local_eval()

    # 更新聚合权重
    def update_update_weight(self):
        for c_id in tqdm(self.client_dict, desc="update update weight"):
            self.client_dict[c_id].cache_keeper.update_raw_eval_list() # 接受到了新模型，更新eval
            self.client_dict[c_id].cache_keeper.update_update_weight(model_dif_adjust=True)

    # 本地训练
    def local(self, turn_on_wandb=True):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_epsilon, total_alpha = 0.0, 0.0
        total_num = 0
        for c_id in tqdm(self.client_dict, desc="local train"):
            if self.args.enable_dp:
                loss, correct, epsilon, alpha = self.client_dict[c_id].local_train()
                total_epsilon += epsilon
                total_alpha += alpha
            else:
                loss, correct = self.client_dict[c_id].local_train()
            total_loss += loss
            total_correct += correct
            total_num += len(self.client_dict[c_id].train_set)

        local_train_acc = total_correct / total_num
        avg_local_train_epsilon, avg_local_train_alpha = total_epsilon / len(self.client_dict), total_alpha / len(self.client_dict)

        if self.args.enable_dp:
            print(f"avg_local_train_epsilon:{avg_local_train_epsilon}, avg_local_train_alpha:{avg_local_train_alpha}")

        print("local_train_loss:{}, local_train_acc:{}".
              format(total_loss, local_train_acc))

        if self.args.turn_on_wandb and turn_on_wandb:
            wandb.log(step=rounds, data={"local_train/loss": total_loss, "local_train/acc": local_train_acc})
            if self.args.enable_dp:
                wandb.log(step=rounds, data={"avg_local_train_epsilon": avg_local_train_epsilon, "avg_local_train_alpha":avg_local_train_alpha})

    def weighted_interpolation_update(self):
        for c_id in tqdm(self.client_dict, desc="calc_new_parameters"):
            self.client_dict[c_id].weighted_model_interpolation_update()

        for c_id in tqdm(self.client_dict, desc="update_parameters"):
            self.client_dict[c_id].model.load_state_dict(self.client_dict[c_id].state_dict)

    # 缓存本地模型，以和下一轮local_train后的模型做区分
    def cache_model(self):
        for c_id in tqdm(self.client_dict, desc="cache_last_local"):
            self.client_dict[c_id].cache_keeper.cache_last_local()

    def cache_received(self):
        for c_id in self.client_dict:
            self.client_dict[c_id].cache_keeper.update_received_memory()

    def clear_received(self):
        for c_id in self.client_dict:
            self.client_dict[c_id].received_model_dict = {}
            self.client_dict[c_id].received_topology_weight_dict = {}
            self.client_dict[c_id].received_w_dict = {}

    # 测试方法不该在这里出现，应该写进基类方法（推荐），或者工具类
    # test local per epoch
    def local_test(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0., 0.
        total_num = 0

        for c_id in tqdm(self.client_dict, desc="local_test"):
            client = self.client_dict[c_id]
            loss, correct = client.local_test()
            total_loss += loss
            total_correct += correct
            # print("client {} contains {} test data".format(c_id, len(client.test_loader)))
            total_num += len(client.test_set)

        avg_acc = total_correct / total_num
        print("local_test_loss:{}, avg_local_test_acc:{}".format(total_loss, avg_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"local_test/loss": total_loss, "local_test/avg_acc": avg_acc})
            if avg_acc > self.best_accuracy:
                wandb.run.summary["best_accuracy"] = avg_acc
                self.best_accuracy = avg_acc

