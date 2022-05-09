import numpy as np


class history(object):

    def __init__(self, args):
        self.args = args
        self.broadcast_weight_history_dict = None
        self.best_accuracy = 0.
        self.train_loss_history = None
        self.train_acc_history = None
        self.test_loss_history = None
        self.test_acc_history = None
        self.broadcast_freq = None

    def get_history_box(self, key):
        if key == "train_loss":
            if self.train_loss_history is None:
                self.train_loss_history = np.zeros(shape=(self.args.comm_round, ))
            return self.train_loss_history
        elif key == "train_acc":
            if self.train_acc_history is None:
                self.train_acc_history = np.zeros(shape=(self.args.comm_round, ))
            return self.train_acc_history
        elif key == "test_loss":
            if self.test_loss_history is None:
                self.test_loss_history = np.zeros(shape=(self.args.comm_round, ))
            return self.test_loss_history
        elif key == "test_acc":
            if self.test_acc_history is None:
                self.test_acc_history = np.zeros(shape=(self.args.comm_round, ))
            return self.test_acc_history
        elif key == "broadcast_weight_history":
            if self.broadcast_weight_history_dict is None:
                self.broadcast_weight_history_dict = {i: np.zeros(shape=(self.args.comm_round, self.args.client_num_in_total)) for i in range(self.args.client_num_in_total)} # broadcast weight history
            return self.broadcast_weight_history_dict
        elif key == "broadcast_freq":
            if self.broadcast_freq is None:
                self.broadcast_freq = np.zeros(shape=(self.args.comm_round, self.args.client_num_in_total, self.args.client_num_in_total))
            return self.broadcast_freq
        else:
            raise NotImplementedError(f"No Such history box:{key}")