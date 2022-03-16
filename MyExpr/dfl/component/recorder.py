import wandb


# todo 入侵性太强了，东一块西一块不知道怎么组织，暂时弃用了，只用来记录拓扑结构等信息
# snapshot of a single iteration
class Snapshot(object):
    def __init__(self, recorder):
        self.recorder = recorder

        self.local_train_loss_per_client = {}
        self.mutual_train_loss_per_client = {}
        self.local_train_correct_per_client = {}

        # 聚合统计数据
        self.mutual_train_regret = 0
        self.local_train_regret = 0
        self.total_local_train_correct = 0

        # 均值
        self.avg_local_train_loss_per_client = 0
        self.avg_mutual_train_loss_per_client = 0
        self.avg_local_train_acc = 0

    def __repr__(self):
        description = "===============================epoch:{:d} iteration:{:d}===============================\n" \
                      "mutual_train_regret:{:f}, local_train_regret:{:f}, avg_local_train_loss_per_client:{:f}\n" \
                      "avg_mutual_train_loss_per_client:{:f}, avg_local_train_acc:{:f}\n" \
                      "=====================================================================================\n"

        return description.format(self.recorder.epoch, self.recorder.iteration, self.mutual_train_regret,
                                  self.local_train_regret, self.avg_local_train_loss_per_client,
                                  self.avg_mutual_train_loss_per_client, self.avg_local_train_acc)

    # 聚合单个iteration的所有记录
    def aggregate(self):
        for c_id in self.recorder.client_dic.keys():
            self.mutual_train_regret += self.mutual_train_loss_per_client[c_id]
            self.local_train_regret += self.local_train_loss_per_client[c_id]
            self.total_local_train_correct += self.local_train_correct_per_client[c_id]

        self.avg_mutual_train_loss_per_client = self.mutual_train_regret / len(self.recorder.client_dic.keys())
        self.avg_local_train_loss_per_client = self.local_train_regret / len(self.recorder.client_dic.keys())
        self.avg_local_train_acc = self.total_local_train_correct / (
                self.recorder.args.batch_size * len(self.local_train_correct_per_client))


class Recorder(object):

    def __init__(self, client_dic, topology_manager, args):
        self.client_dic = client_dic
        self.topology_manager = topology_manager
        self.args = args

        self.epoch = 0
        self.iteration = 0
        self.cur_snapshot = Snapshot(self)
        self.train_history_per_iteration = []
        self.train_history_per_epoch = []
        self.print_log = False
        self.wandb_log = True

    # iteration 其实应该放进Snapshot里
    def record_local_train_loss(self, c_id, loss):
        self.cur_snapshot.local_train_loss_per_client[c_id] = loss

    def record_mutual_train_loss(self, c_id, loss):
        self.cur_snapshot.mutual_train_loss_per_client[c_id] = loss

    def record_local_train_correct(self, c_id, correct):
        self.cur_snapshot.local_train_correct_per_client[c_id] = correct

    def print_round_log(self):
        print(self.cur_snapshot)

    def next_iteration(self):
        self.cur_snapshot.aggregate()
        if self.print_log:
            self.print_round_log()
        self.iteration += 1
        self.train_history_per_iteration.append(self.cur_snapshot)
        self.cur_snapshot = Snapshot(self)

    # epoch
    def record_local_test(self, avg_local_test_loss, avg_local_test_acc):
        if self.print_log:
            print("avg_local_test_loss:{:f} avg_local_test_acc:{:f}\n ".format(avg_local_test_loss, avg_local_test_acc))
        if self.wandb_log:
            wandb.log({"avg_local_test_loss": avg_local_test_loss, "avg_local_test_acc": avg_local_test_acc})

    def record_overall_test(self, avg_overall_test_loss, avg_overall_test_acc):
        if self.print_log:
            print("avg_overall_test_loss:{:f} avg_overall_test_acc:{:f}\n ".format(avg_overall_test_loss,
                                                                                   avg_overall_test_acc))
        if self.wandb_log:
            wandb.log({"avg_overall_test_loss": avg_overall_test_loss, "avg_overall_test_acc": avg_overall_test_acc})

    def next_epoch(self):
        self.epoch += 1
        self.iteration = 0

        # todo 优化epoch总结输出
        epoch_mutual_train_regret = 0
        epoch_local_train_correct = 0
        epoch_local_train_regret = 0
        for snapshot in self.train_history_per_iteration:
            epoch_mutual_train_regret += snapshot.mutual_train_regret
            epoch_local_train_correct += snapshot.total_local_train_correct
            epoch_local_train_regret += snapshot.local_train_regret

        # 记录本轮数据
        self.train_history_per_epoch.append(self.train_history_per_iteration)

        epoch_local_train_acc = epoch_local_train_correct / (len(self.train_history_per_iteration)
                                                             * self.args.batch_size * len(self.client_dic))
        avg_epoch_mutual_train_regret = epoch_mutual_train_regret / (len(self.train_history_per_iteration)
                                                                     * len(self.client_dic))
        avg_epoch_local_train_regret = epoch_local_train_regret / (len(self.train_history_per_iteration)
                                                                   * len(self.client_dic))
        if self.print_log:
            print("avg_epoch_mutual_train_regret:{:f} avg_epoch_local_train_regret:{:f}"
                  " epoch_mutual_train_acc:{:f}\n ".format(avg_epoch_mutual_train_regret, avg_epoch_local_train_regret,
                                                           epoch_local_train_acc))

        # 重置轮记录
        self.train_history_per_iteration = []

        if self.wandb_log:
            wandb.log({"avg_epoch_mutual_train_regret" : avg_epoch_mutual_train_regret,
                       "avg_epoch_local_train_regret": avg_epoch_local_train_regret,
                       "epoch_mutual_train_acc": epoch_local_train_acc})
