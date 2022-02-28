import csv

import numpy as np


class DataLoader(object):

    def __init__(self,  data_path):
        self.data_path = data_path
        # len(SUSY) = 5,000,000
        # The last 500,000 examples are used as a test set.n about your data set.
        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

    def load_data(self):
        with open(self.data_path) as susyfile:
            readCSV = csv.reader(susyfile, delimiter=",")
            for i, row in enumerate(readCSV):
                if i < 4500000:
                    self.train_X.append(np.asarray(row[1:], dtype=np.float32))
                    self.train_Y.append(int(row[0].split('.')[0]))
                else:
                    self.test_X.append(np.asarray(row[1:], dtype=np.float32))
                    self.test_Y.append(int(row[0].split('.')[0]))
        return self.train_X, self.train_Y, self.test_X, self.test_Y
