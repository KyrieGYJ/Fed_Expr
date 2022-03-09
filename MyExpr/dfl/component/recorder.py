
# snapshot of a single iteration
class snapshot(object):
    def __init__(self, recorder):
        self.recorder = recorder
        self.epoch = self.recorder.epoch
        self.iteration = self.recorder.iteration

        self.local_train_loss_per_client = {}
        self.mutual_train_loss_per_client = {}

        self.mutual_train_regret = 0
        self.local_train_regret = 0
        self.avg_local_train_loss_per_client = 0
        self.avg_mutual_train_loss_per_client = 0
        self.avg_train_loss_per_batch = 0

    def __repr__(self):
        description = "===============================epoch:{:s} iteration:{:s}===============================\n" \
                      "local_train_loss_per_client:{:s} \n mutual_train_loss_per_client:{:s} \n " \
                      "avg_train_loss_per_batch:{:d} \n mutual_train_regret:{:d} \n " \
                        "local_train_regret:{:d} \n avg_local_train_loss_per_client:{:d} \n" \
                        "avg_mutual_train_loss_per_client:{:d}\n" \
                        "=====================================================================================\n"

        return description.format(self.epoch, self.iteration, self.local_train_loss_per_client, self.mutual_train_loss_per_client,
                          self.avg_train_loss_per_batch, self.mutual_train_regret,
                          self.local_train_regret, self.avg_local_train_loss_per_client,
                          self.avg_mutual_train_loss_per_client)

    def aggregate(self):
        for c_id in self.recorder.client_dic.keys():
            self.mutual_train_regret += self.mutual_train_loss_per_client[c_id]
            self.local_train_regret += self.local_train_loss_per_client[c_id]

        self.avg_local_train_loss_per_client = self.avg_local_train_loss_per_client / len(self.recorder.client_dic.keys())
        self.avg_mutual_train_loss_per_client = self.mutual_train_regret / len(self.recorder.client_dic.keys())
        self.avg_train_loss_per_batch = self.avg_local_train_loss_per_client / self.recorder.args.batch_size


class Recorder(object):

    def __init__(self, client_dic, topology_manager, args):
        self.client_dic = client_dic
        self.topology_manager = topology_manager
        self.args = args

        self.epoch = 0
        self.iteration = 0
        self.cur_snapshot = snapshot(self)
        self.train_history_per_iteration = []
        self.train_history_per_epoch = []

        self.print_log = True

    def next_iteration(self):
        self.cur_snapshot.aggregate()
        if self.print_log:
            self.print_round_log()
        self.train_history_per_iteration.append(self.cur_snapshot)
        self.cur_snapshot = snapshot(self)

    def next_epoch(self):
        self.epoch += 1
        self.iteration = 0

        epoch_mutual_train_regret = 0
        for ss in self.train_history_per_iteration:
            epoch_mutual_train_regret += ss.mutual_train_regret
        if self.print_log:
            print("epoch_mutual_train_regret:{:d}".format(epoch_mutual_train_regret))

    def record_local_train_loss(self, c_id, loss):
        self.cur_snapshot.local_train_loss_per_client[c_id] = loss

    def record_mutual_train_loss(self, c_id, loss):
        self.cur_snapshot.mutual_train_loss_per_client[c_id] = loss

    def print_round_log(self):
        print(self.cur_snapshot)