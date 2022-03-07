import csv

import numpy as np


class DataLoader(object):

    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        # len(SUSY) = 5,000,000
        # The last 500,000 examples are used as a test set.n about your data set.
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

    def load_data(self):
        with open(self.data_path) as susyfile:
            readCSV = csv.reader(susyfile, delimiter=",")
            cur_batch = 0
            cur_X = []
            cur_Y = []
            for i, row in enumerate(readCSV):
                cur_batch += 1
                cur_X.append(np.asarray(row[1:], dtype=np.float32))
                cur_Y.append(int(row[0].split('.')[0]))
                if cur_batch < self.batch_size:
                    continue
                if i < 4500000:
                    self.train_X.append(cur_X)
                    self.train_Y.append(cur_Y)
                else:
                    self.test_X.append(cur_X)
                    self.test_Y.append(cur_Y)
                cur_X = []
                cur_Y = []
        return self.train_X, self.train_Y, self.test_X, self.test_Y
