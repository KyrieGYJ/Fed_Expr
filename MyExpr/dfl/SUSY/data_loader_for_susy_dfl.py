import csv

import numpy as np
from sklearn.cluster import KMeans


class DataLoader(object):
    def __init__(self, data_name, data_path, client_id_list, train_sample_num_in_total, beta):
        # SUSY
        self.data_name = data_name
        self.data_path = data_path
        self.client_list = client_id_list
        self.sample_num_in_total = train_sample_num_in_total
        # beta是指作为iid数据的比例，通过K-means聚类实现非独立同分布。
        self.beta = beta
        self.streaming_full_dataset_X = []
        self.streaming_full_dataset_Y = []
        # {client_id : dataset}
        self.StreamingDataDict = {}
        # {client_id : {'x' : [[batch1], [batch2], ...}, 'y':[[batch1], ...]}
        # batch1 -> 'x':[1,2,3,4,5,...,M]; "y":0
        self.batch_size = 100
        self.BatchDataDict = {}
        self.full_test_X = []
        self.full_test_Y = []
        self.test_X = []
        self.test_Y = []

    """
        return streaming_data
            key: client_id
            value: [sample1, sample2, ..., sampleN]
                    sample: {"x": [1,2,3,4,5,...,M]; "y":0}
    """

    def load_data(self):
        # 要注意不要读超了，默认训练数据集大小为4,500,000
        print("加载全部数据...", end="")
        self.preprocessing()
        print("加载对抗数据...", end="")
        self.load_adversarial_data()
        print("加载随机数据...", end="")
        self.load_stochastic_data()
        # return self.StreamingDataDict
        self.compose_to_batch(100)
        return self.BatchDataDict


    # beta (clustering, GMM)
    def load_adversarial_data(self):
        streaming_data = self.read_csv_file_for_cluster(self.beta)
        print("完成")
        return streaming_data

    def load_stochastic_data(self):
        streaming_data = self.read_csv_file(self.beta)
        print("完成")
        return streaming_data

    def read_csv_file(self, percent):
        iteration_number = int(self.sample_num_in_total / len(self.client_list))
        index_start = int(percent * self.sample_num_in_total)
        stochastic_data_x = []
        stochastic_data_y = []

        # 加载所有随机数据(除去聚类的部分)
        while index_start < len(self.streaming_full_dataset_X):
            stochastic_data_x.append(self.streaming_full_dataset_X[index_start])
            stochastic_data_y.append(self.streaming_full_dataset_Y[index_start])
            index_start += 1

        # 去除超出iteration_num的部分(保持每个client样本数稳定)
        for c_index in self.client_list:
            # todo 改了self.client_list[c_index] -> c_index，出错记得check
            if len(self.StreamingDataDict[c_index]) > iteration_number:
                iteration_number_cache = iteration_number
                while iteration_number_cache < len(self.StreamingDataDict[c_index]):
                    stochastic_data_x.append(self.StreamingDataDict[c_index][iteration_number_cache]['x'])
                    stochastic_data_y.append(self.StreamingDataDict[c_index][iteration_number_cache]['y'])
                    iteration_number_cache += 1
                self.StreamingDataDict[c_index] = self.StreamingDataDict[c_index][0:iteration_number]

        # 把多余的数据填充到缺少数据的client里面
        client_index = 0
        full_count = 0
        for i in range(len(stochastic_data_x)):
            while len(self.StreamingDataDict[client_index]) == iteration_number:
                client_index += 1
                full_count += 1
            sample = {}
            sample["x"] = stochastic_data_x[i]
            sample["y"] = stochastic_data_y[i]
            self.StreamingDataDict[self.client_list[client_index]].append(sample)
            # todo 条件 full_count == len(self.client_list) - 1 是否多余
            if len(self.StreamingDataDict[self.client_list[client_index]]) == iteration_number and \
                    full_count == len(self.client_list) - 1:
                full_count += 1
            # todo 终止条件放这是不是有点臭
            if full_count == len(self.client_list):
                break
        return self.StreamingDataDict

    def read_csv_file_for_cluster(self, percent):
        data = []
        label = []
        for client_id in self.client_list:
            self.StreamingDataDict[client_id] = []
        if percent == 0:
            return self.StreamingDataDict

        for i, row in enumerate(self.streaming_full_dataset_X):
            if i >= (self.sample_num_in_total * percent):
                break
            data.append(self.streaming_full_dataset_X[i])
            label.append(self.streaming_full_dataset_Y[i])
        clusters = self.kMeans(data)

        for i, cluster in enumerate(clusters):
            sample = {}
            sample["y"] = label[i]
            sample["x"] = data[i]
            self.StreamingDataDict[self.client_list[cluster]].append(sample)
        return self.StreamingDataDict

    def kMeans(self, X):
        kmeans = KMeans(n_clusters=len(self.client_list))
        kmeans.fit(X)
        return kmeans.labels_

    def preprocessing(self):
        with open(self.data_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(readCSV):
                x, y = np.asarray(row[1:], dtype=np.float32), int(row[0].split('.')[0])
                if i < self.sample_num_in_total:
                    self.streaming_full_dataset_X.append(x)
                    self.streaming_full_dataset_Y.append(y)
                elif i >= 4500000:
                    self.full_test_X.append(x)
                    self.full_test_Y.append(y)
        print("完成")

    def compose_to_batch(self, batch_size):
        # 100应该差不多
        for c_index in self.client_list:
            self.BatchDataDict[c_index] = {}
            self.BatchDataDict[c_index]['x'], self.BatchDataDict[c_index]['y'] = [], []
            cur_X, cur_Y= [], []
            for sample in self.StreamingDataDict[c_index]:
                cur_X.append(sample['x'])
                cur_Y.append(sample['y'])
                if len(cur_X) == batch_size:
                    self.BatchDataDict[c_index]['x'].append(cur_X)
                    self.BatchDataDict[c_index]['y'].append(cur_Y)
                    cur_X = []
                    cur_Y = []

        cur_X, cur_Y = [], []
        for x, y in zip(self.full_test_X, self.full_test_Y):
            cur_X.append(x)
            cur_Y.append(y)
            if len(cur_X) == batch_size:
                self.test_X.append(cur_X)
                self.test_Y.append(cur_Y)
                cur_X = []
                cur_Y = []




