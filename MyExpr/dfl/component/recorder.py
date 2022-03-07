

class snapshot(object):
    def __init__(self, epoch):
        self.epoch = epoch
        self.local_train_loss_per_client = {}
        self.mutual_train_loss_per_client = {}
        self.avg_train_loss_per_batch = []
        self.regret = 0
        # todo avg_local_train_loss_per_client
        # todo avg_mutual_train_loss_per_client
        # todo regret_per_batch

    def __repr__(self):
        description = "local_train_loss_per_client:{:s} \n mutual_train_loss_per_client:{:s} \n " \
                      "avg_train_loss_per_batch:{:s} \n regret:{:s} \n "
        return description.format(self.local_train_loss_per_client, self.mutual_train_loss_per_client,
                                  self.avg_train_loss_per_batch, self.regret)


class Recorder(object):

    def __init__(self, client_dic, topology_manager):
        self.client_dic = client_dic
        self.topology_manager = topology_manager
        self.cur_snapshot = snapshot(epoch=0)
        self.train_history = []

        self.print_log = True

    def next_iteration(self):
        if self.print_log:
            self.print_round_log()
        self.train_history.append(self.cur_snapshot)
        self.cur_snapshot = snapshot(self.train_history[-1].epoch + 1)
        # for c_id in self.client_dic.keys():
        #     self.cur_snapshot.local_train_loss_per_client[c_id] = []

    def record_local_train_loss(self, c_id, loss):
        self.cur_snapshot.local_train_loss_per_client[c_id] = loss

    def record_mutual_train_loss(self, c_id, loss):
        self.cur_snapshot.mutual_train_loss_per_client[c_id] = loss

    def record_regret(self):
        for c_id in self.client_dic.keys():
            self.cur_snapshot.regret += self.cur_snapshot.local_train_loss_per_client[c_id]

    def print_round_log(self):
        print(self.cur_snapshot)