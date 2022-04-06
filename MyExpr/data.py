import torch
import os.path
from torchvision.datasets import utils, MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from PIL import Image
import numpy as np
from os.path import join, exists
import torchvision

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from MyExpr.utils import compute_emd
from MyExpr.utils import print_debug
from MyExpr.latent_dist import get_embeddings

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10

class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        utils.makedir_exist_ok(self.raw_folder)
        utils.makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)


def Dataset(args):
    trainset, testset = None, None

    if args.dataset == 'cifar10':
        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # len(trainset) = 50000 len(testset) = 10000
        trainset = CIFAR10(root=args.data_dir, train=True, download=True, transform=tra_trans)
        testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=val_trans)

    if args.dataset == 'femnist' or 'mnist':
        tra_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if args.dataset == 'femnist':
            trainset = FEMNIST(root=args.data_dir, train=True, transform=tra_trans)
            testset = FEMNIST(root=args.data_dir, train=False, transform=val_trans)
        if args.dataset == 'mnist':
            trainset = MNIST(root=args.data_dir, train=True, transform=tra_trans)
            testset = MNIST(root=args.data_dir, train=False, transform=val_trans)

    return trainset, testset


class Data(object):

    def __init__(self, args):
        self.args = args
        self.train_data, self.test_data = Dataset(args)
        self.train_all = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.test_all = DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.train_loader, self.validation_loader, self.test_loader, self.test_non_iid = [], [], [], []

        self.generate_loader = None
        self.train_idx_dict = {}

        # 记录每个client持有哪些类
        self.client_class_dic = {}
        # 记录每个类被哪些client持有
        self.class_client_dic = {}

        # 记录每个client属于哪个分布
        self.client_dist_dict = {}
        # 记录每个分布包含哪些client
        self.dist_client_dict = {}

        # 兼容non_iid_latent2
        # ImageNet normalization constants
        imagenet_normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))

        self.setup_transform = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   imagenet_normalize])

        self.setup_test_transform = transforms.Compose([transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        imagenet_normalize])


        if args.data_distribution == "iid":
            self.generate_loader = self.iid
        elif args.data_distribution == "non-iid_pathological":
            self.generate_loader = self.non_iid_pathological
        elif args.data_distribution == "non-iid_pathological2":
            self.generate_loader = self.non_iid_pathological2
        elif args.data_distribution == "non-iid_latent":
            self.generate_loader = self.non_iid_latent
        elif args.data_distribution == "non-iid_latent2":
            self.generate_loader = self.non_iid_latent_decrapted

        self.shuffle = True

    # todo 改编自LG-FedAvg，开源前加引用
    def non_iid_pathological(self):
        train_data = self.train_data
        test_data = self.test_data
        args = self.args

        # dict_users_train 记录了每个用户持有的数据下标集合 (dict的key是按从小到大排序的)
        dict_users_train, rand_set_all = self.client_noniid(train_data, args.client_num_in_total, args.shards_per_user,
                                                            rand_set_all=[], args=args)
        dict_users_test, rand_set_all = self.client_noniid(test_data, args.client_num_in_total, args.shards_per_user,
                                                           rand_set_all=rand_set_all, args=args)
        # for key in dict_users_train:
        #     print(key)
        # for key in dict_users_test:
        #     print(key)

        self.train_idx_dict = dict_users_train
        class_test_dataset = []
        # 获得每个标签对应的测试数据
        dic_test_data = self.data_per_label(test_data)
        # 按顺序取，确保class_test_dataset下标和label能对应上
        for i in range(len(dic_test_data)):
            class_test_dataset.append(torch.utils.data.Subset(test_data, indices=dic_test_data[i]))
            self.test_non_iid.append(
                DataLoader(class_test_dataset[i], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))

        # 统计每个类分别被哪些client持有
        for ix in dict_users_train:
            self.client_class_dic[ix] = np.unique(torch.tensor(train_data.targets)[dict_users_train[ix]])

            # client记入字典 {label:client_id}
            for label in self.client_class_dic[ix]:
                if label not in self.class_client_dic:
                    self.class_client_dic[label] = []
                self.class_client_dic[label].append(ix)

        # 划分validation
        client_train_dataset, client_validation_dataset = self.validation_partition(dict_users_train, train_data, args)
        client_test_dataset = []

        # 生成每个clietn的train, validation, test, non-iid的dataloader
        for client_id in range(args.client_num_in_total):
            client_test_dataset.append(torch.utils.data.Subset(test_data, indices=dict_users_test[client_id]))

            self.train_loader.append(
                DataLoader(client_train_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))
            self.validation_loader.append(
                DataLoader(client_validation_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))
            self.test_loader.append(
                DataLoader(client_test_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))

            # 验证
            train_class = np.unique(torch.tensor(train_data.targets)[dict_users_train[client_id]])
            test_class = np.unique(torch.tensor(test_data.targets)[dict_users_test[client_id]])
            train_class.sort()
            test_class.sort()
            assert((train_class == test_class).all())
            print(f"client {client_id} has {len(client_train_dataset[client_id])} train data, "
                  f"{len(client_validation_dataset[client_id])} validation data and {len(client_test_dataset[client_id])} test data")

        average_emd = np.mean(
            [compute_emd([train_data.targets[x] for x in dict_users_train[ix]], train_data.targets) for ix in
             dict_users_train], axis=0)
        print(f'> Global mean EMD: {average_emd}')

    def non_iid_pathological2(self):
        train_data = self.train_data
        test_data = self.test_data
        args = self.args

        dict_users_train, dict_users_label = self.pathological2_helper(train_data)
        dict_users_test, dict_users_label = self.pathological2_helper(test_data)

        # for i in range(args.client_num_in_total):
        #     print(f"client {i} has label: {dict_users_label[i]}")

        self.train_idx_dict = dict_users_train

        class_test_dataset = []

        # 获得每个标签对应的测试数据
        dic_test_data = self.data_per_label(test_data)

        # 按顺序取，确保class_test_dataset下标和label能对应上
        for i in range(len(dic_test_data)):
            class_test_dataset.append(torch.utils.data.Subset(test_data, indices=dic_test_data[i]))
            self.test_non_iid.append(
                DataLoader(class_test_dataset[i], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))

        # 统计每个类分别被哪些client持有
        for ix in dict_users_train:
            self.client_class_dic[ix] = np.unique(torch.tensor(train_data.targets)[dict_users_train[ix]])

            # client记入字典 {label:client_id}
            for label in self.client_class_dic[ix]:
                if label not in self.class_client_dic:
                    self.class_client_dic[label] = []
                self.class_client_dic[label].append(ix)

            print(f"{dict_users_label[ix]} == {self.client_class_dic[ix]}")

        # 划分validation
        client_train_dataset, client_validation_dataset = self.validation_partition(dict_users_train, train_data, args)
        client_test_dataset = []

        # 生成每个clietn的train, validation, test, non-iid的dataloader
        for client_id in range(args.client_num_in_total):
            client_test_dataset.append(torch.utils.data.Subset(test_data, indices=dict_users_test[client_id]))

            self.train_loader.append(
                DataLoader(client_train_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))
            self.validation_loader.append(
                DataLoader(client_validation_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))
            self.test_loader.append(
                DataLoader(client_test_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))

            # 验证
            train_class = np.unique(torch.tensor(train_data.targets)[dict_users_train[client_id]])
            test_class = np.unique(torch.tensor(test_data.targets)[dict_users_test[client_id]])
            train_class.sort()
            test_class.sort()
            assert ((train_class == test_class).all())
            print(f"client {client_id} has {len(client_train_dataset[client_id])} train data, "
                  f"{len(client_validation_dataset[client_id])} validation data and {len(client_test_dataset[client_id])} test data")

        average_emd = np.mean(
            [compute_emd([train_data.targets[x] for x in dict_users_train[ix]], train_data.targets) for ix in
             dict_users_train], axis=0)
        print(f'> Global mean EMD: {average_emd}')

    def pathological2_helper(self, dataset):
        args = self.args
        label_idx_dict = self.data_per_label(dataset)

        shard_per_class = args.num_clients_per_dist
        num_classes = len(np.unique(dataset.targets))
        class_per_distribution = int(num_classes / args.num_distributions)

        # 生成 label -> shards的字典
        for label in label_idx_dict.keys():
            x = label_idx_dict[label]
            # 计算分shard后多余的数据量
            num_leftover = len(x) % shard_per_class
            # 记录余数
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            # 裁剪多余数据
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            # reshape成[shard_num, shard_size]
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            # 多余数据有限填充到x[i]
            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            # x 为 [shard_num, shard_zie]
            label_idx_dict[label] = x

        client_data_dict = {i: None for i in range(args.client_num_in_total)}
        client_label_dict = {i: None for i in range(args.client_num_in_total)}

        client_id = 0
        # 为每个dist中的每个client分配shard
        for dist_id in range(args.num_distributions):
            for i in range(args.num_clients_per_dist):
                idx_set = []
                label_set = []
                for j in range(class_per_distribution):
                    label = dist_id * 2 + j
                    label_set.append(label)
                    # replace是指抽取不放回，但是这里并没有指定size，每次都独立取出一个idx。
                    idx = np.random.choice(len(label_idx_dict[label]), replace=False)
                    # 弹出idx，防止重复取
                    idx_set.append(label_idx_dict[label].pop(idx))
                client_data_dict[client_id] = np.concatenate(idx_set)
                client_label_dict[client_id] = label_set
                client_id += 1

        # check
        test = []
        for key, value in client_data_dict.items():
            label_num = np.unique(torch.tensor(dataset.targets)[value])
            assert (len(label_num)) <= class_per_distribution
            test.append(value)
        test = np.concatenate(test)
        assert (len(test) == len(dataset))
        assert (len(set(list(test))) == len(dataset))

        return client_data_dict, client_label_dict

    def non_iid_latent(self):
        train_data, test_data, args = self.train_data, self.test_data, self.args
        train_idx_dict, test_idx_dict = self.partition_data(args.client_num_in_total, alpha=0.5)
        Y_train = np.array(self.train_data.targets)

        self.train_idx_dict = train_idx_dict

        # todo check字典key是顺序的

        client_train_dataset, client_validation_dataset = self.validation_partition(train_idx_dict, train_data, args)
        client_test_dataset = []

        class_num = len(np.unique(torch.tensor(train_data.targets)))
        client_distributions = []
        # collect client distributions
        for client_id in range(args.client_num_in_total):
            client_train_idx = train_idx_dict[client_id]
            unq, unq_cnt = np.unique(Y_train[client_train_idx], return_counts=True)
            distribution = [0] * class_num
            for label in range(len(unq)):
                distribution[label] = unq_cnt[label]
            client_distributions.append(distribution)
            # print(f"client {client_id} has label {unq}")
        # print(f'client has label distributions are as follow: \n {client_distributions}')

        print("clustering for different distribution...", end=" ")
        # clustering
        kmeans = KMeans(n_clusters=args.num_distributions)
        kmeans.fit(client_distributions)
        distributions = kmeans.labels_
        print("done")
        distributions_dict = {}
        # todo 要改
        self.dist_client_dict = {}
        self.client_dist_dict = {}
        for client_id, cluster_id in enumerate(distributions):
            if cluster_id not in distributions_dict:
                distributions_dict[cluster_id] = []
                self.dist_client_dict[cluster_id] = []
            self.dist_client_dict[cluster_id].append(client_id)
            self.client_dist_dict[client_id] = cluster_id
            distributions_dict[cluster_id] += test_idx_dict[client_id]
            # todo 还得记录训练数据集作为上界
            # distributions_dict[cluster_id] += train_idx_dict[client_id]

        distribution_test_dataset = []
        for i in range(len(distributions_dict)):
            distribution_test_dataset.append(torch.utils.data.Subset(test_data, indices=distributions_dict[i]))
            print(f"distribution {i} contains {self.dist_client_dict[i]}")
            self.test_non_iid.append(
                DataLoader(distribution_test_dataset[i], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))

        # 生成dataloader
        for client_id in range(args.client_num_in_total):
            client_test_dataset.append(torch.utils.data.Subset(self.test_data, indices=test_idx_dict[client_id]))
            self.train_loader.append(DataLoader(client_train_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))
            self.validation_loader.append(DataLoader(client_validation_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.num_workers))
            self.test_loader.append(DataLoader(client_test_dataset[client_id], batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers))
            # 验证
            train_class = np.unique(torch.tensor(train_data.targets)[train_idx_dict[client_id]])
            test_class = np.unique(torch.tensor(test_data.targets)[test_idx_dict[client_id]])
            train_class.sort()
            test_class.sort()
            # print(client_id, train_class, test_class)
            # todo 在latent noniid情况下分local test这种操作估计是有问题的，因为test的类和train的类会有不重叠
            # assert ((train_class == test_class).all())
            print(f"client {client_id} has {len(client_train_dataset[client_id])} train data, "
                  f"{len(client_validation_dataset[client_id])} validation data and {len(client_test_dataset[client_id])} test data")

        average_emd = np.mean(
            [compute_emd([train_data.targets[x] for x in train_idx_dict[ix]], train_data.targets)
             for ix in range(args.client_num_in_total)], axis=0)
        print(f'> Global mean EMD: {average_emd}')

    # todo 加上validation划分
    def iid(self):
        trainset = self.train_data
        testset = self.test_data
        args = self.args

        # 均等分训练数据集
        num_train = [int(len(trainset) / args.client_num_in_total) for _ in range(args.client_num_in_total)]
        # 求前缀和
        cumsum_train = torch.tensor(list(num_train)).cumsum(dim=0).tolist()
        num_test = [int(len(testset) / args.client_num_in_total) for _ in range(args.client_num_in_total)]
        cumsum_test = torch.tensor(list(num_test)).cumsum(dim=0).tolist()

        # 默认iid
        # 下标数组，下同
        idx_train = range(len(trainset.targets))
        idx_test = range(len(testset.targets))
        # if args.data_distribution == "non-iid(1)":
        #     # 将trainset的下标按标签进行排序，然后放到idx_train， 下同
        #     idx_train = sorted(range(len(trainset.targets)), key=lambda k: trainset.targets[k])  # split by class
        #     idx_test = sorted(range(len(testset.targets)), key=lambda k: testset.targets[k])  # split by class

        # 将划分好的下标，按照num_train划分成不同子集，【前缀和-num_train: 前缀和】
        splited_trainset = [Subset(trainset, idx_train[off - l:off]) for off, l in zip(cumsum_train, num_train)]
        splited_testset = [Subset(testset, idx_test[off - l:off]) for off, l in zip(cumsum_test, num_test)]

        self.train_all = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.test_all = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        self.train_loader = [
            DataLoader(splited_trainset[i], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            for i in range(args.client_num_in_total)]

        self.test_loader = [
            DataLoader(splited_testset[i], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            for i in range(args.client_num_in_total)]
        # self.test_loader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    # 将dataset中的数据（下标）放入字典 {label : index}
    def data_per_label(self, dataset):
        idxs_dic = {}

        for i in range(len(dataset)):
            label = torch.tensor(dataset.targets[i]).item()
            if label not in idxs_dic.keys():
                idxs_dic[label] = []
            idxs_dic[label].append(i)
        return idxs_dic

    def client_noniid(self, dataset, num_users, shard_per_user, rand_set_all, args):
        """
        Sample non-IID client data from dataset in pathological manner - from LG-FedAvg implementation
        :param dataset:
        :param num_users:
        :return: (dictionary, where keys = client_id / index, and values are dataset indices), rand_set_all (all classes)

        shard_per_user should be a factor of the dataset size
        """
        seed = args.data_seed
        # 确保字典key顺序
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

        idxs_dict = self.data_per_label(dataset)

        # 统计标签种类
        num_classes = len(np.unique(dataset.targets))
        # 计算每个类要分出多少个shard
        shard_per_class = int(shard_per_user * num_users / num_classes)

        # 生成 label -> shards的字典
        for label in idxs_dict.keys():
            x = idxs_dict[label]
            # 计算分shard后多余的数据量
            num_leftover = len(x) % shard_per_class
            # 记录余数
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            # 裁剪多余数据
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            # reshape成[shard_num, shard_size]
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            # 多余数据有限填充到x[i]
            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            # x 为 [shard_num, shard_zie]
            idxs_dict[label] = x

        # 总共num_classes*shard_per_class个shard， 随机给每个client分配
        np.random.seed(seed)
        if len(rand_set_all) == 0:
            rand_set_all = list(range(num_classes)) * shard_per_class
            np.random.shuffle(rand_set_all)
            rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

        # 按照上述shard划分，为每个user分配对应的shard
        # Divide and assign
        np.random.seed(seed)
        for i in range(num_users):
            rand_set_label = rand_set_all[i]
            rand_set = []
            for label in rand_set_label:
                # replace是指抽取不放回，但是这里并没有指定size，每次都独立取出一个idx。
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                # 弹出idx，防止重复取
                rand_set.append(idxs_dict[label].pop(idx))
            # 将rand_set拼接成一条数据，放入dic_user[i]中
            dict_users[i] = np.concatenate(rand_set)

        # 校验划分结果
        test = []
        for key, value in dict_users.items():
            # 抽出当前client包含的class
            x = np.unique(torch.tensor(dataset.targets)[value])
            assert (len(x)) <= shard_per_user
            test.append(value)
        test = np.concatenate(test)
        assert (len(test) == len(dataset))
        assert (len(set(list(test))) == len(dataset))

        return dict_users, rand_set_all

    def validation_partition(self, dict_users_train, train_data, args):
        client_train_dataset, client_validation_dataset = [], []
        # 划分validation
        for ix in range(args.client_num_in_total):
        # for ix in dict_users_train:
            client_dataset = torch.utils.data.Subset(train_data, indices=dict_users_train[ix])

            # 划分validation
            len_train_split = int(np.round(args.train_split * len(client_dataset)))
            len_val_split = len(client_dataset) - len_train_split

            torch.manual_seed(args.data_seed)
            split_lens = [len_train_split, len_val_split]
            datasets = torch.utils.data.random_split(client_dataset, split_lens)
            client_train_dataset.append(datasets[0])
            client_validation_dataset.append(datasets[1])

        return client_train_dataset, client_validation_dataset

    # todo 引用FedML
    def partition_data(self, n_nets, alpha):
        print("*********partition data***************")
        X_train, y_train = self.train_data.data, np.array(self.train_data.targets)
        X_test, y_test = self.test_data.data, np.array(self.test_data.targets)

        min_size = 0
        K = len(np.unique(self.train_data.targets))
        train_N, test_N = len(y_train), len(y_test)
        print(f"train_N = {train_N}, test_N = {test_N}, K = {K}")

        train_idx_dict = {}
        test_idx_dict = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            test_idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]
                # print(train_idx_k, test_idx_k)
                # print(y_train[train_idx_k], y_test[test_idx_k])
                np.random.shuffle(train_idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # Balance
                # 相当于 p * 1 if len(idx_j) < (N / n_nets) else 0
                # 也就是如果client[j]的数据量已经大于平均值时，不再分配proportions。
                train_proportions = np.array([p * (len(idx_j) < train_N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                # 这里有可能某个label train分到了但是test没分到，
                test_proportions = np.array([p * (len(idx_j) < test_N / n_nets) for p, idx_j in zip(proportions, test_idx_batch)])

                # 正规化
                train_proportions = train_proportions / train_proportions.sum()
                test_proportions = test_proportions / test_proportions.sum()

                # print(f"train_proportions:{train_proportions} \n test_proportions:{test_proportions}")

                # 求proportions的前缀和，并且放大len(idx_k)倍
                # cumsum最后一项是总数，删除掉
                train_proportions = (np.cumsum(train_proportions) * len(train_idx_k)).astype(int)[:-1]
                test_proportions = (np.cumsum(test_proportions) * len(test_idx_k)).astype(int)[:-1]

                # print(train_proportions, test_proportions)

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(train_idx_k, train_proportions))]
                test_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(test_idx_batch, np.split(test_idx_k, test_proportions))]

                min_size = min([len(idx_j) for idx_j in idx_batch])
            #     print(idx_batch)
            # print(min_size)

        # 确保了顺序
        # 把batch数据记录到字典中 {client_id: idx_batch}
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            train_idx_dict[j] = idx_batch[j]
            test_idx_dict[j] = test_idx_batch[j]

        return train_idx_dict, test_idx_dict

    # 统计每个client中，每个标签数量
    def record_net_data_stats(self, label, idx_dict):
        net_cls_counts = {}

        for net_i, dataidx in idx_dict.items():
            unq, unq_cnt = np.unique(label[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        print('Data statistics: %s' % str(net_cls_counts))
        return net_cls_counts

    def non_iid_latent_decrapted(self):
        # args = self.args
        #
        # self.precomputed_root = "./../data/precomputed"
        # self.embedding_fname = 'embeddings-d=cifar10-e=4-st=5-lr=0.001-mo=0.9-wd=0.0001-fc2-train.npy'
        # self.train_embedding_fname = self.embedding_fname
        # self.test_embedding_fname = 'embeddings-d=cifar10-e=4-st=5-lr=0.001-mo=0.9-wd=0.0001-fc2-test.npy'
        # kmeans_labels_prefix = f'kmeans-nd={args.num_distributions}-s={args.seed}-ds={args.data_seed}'
        # self.kmeans_train_fname = f'{kmeans_labels_prefix}-{self.embedding_fname}'
        # self.kmeans_test_fname = f'{kmeans_labels_prefix}-{self.test_embedding_fname}'
        #
        # """
        # 参考FedFomo
        # Initialize client data distributions with through latent non-IID method
        # - Groups datapoints into D groupings based on clustering their
        #   hidden-layer representations from a pre-trained model
        # """
        # # init_distribution
        # dict_data = {'inputs': np.array(self.train_data.data),
        #              'targets': np.array(self.train_data.targets),
        #              'test_inputs': np.array(self.test_data.data),
        #              'test_targets': np.array(self.test_data.targets)}
        #
        # # 先计算VGG在这个数据集上的训练后的，倒数第二个隐藏层输出作为model embedding，随后根据emb edding聚类
        # # 如果已有计算好的embedding，直接读取
        # try:
        #     path = join(self.precomputed_root, self.train_embedding_fname)
        #     print(f'> Loading training embeddings from {path}...')
        #     with open(path, 'rb') as f:
        #         train_embeddings = np.load(f)
        #         dict_data['train_embeddings'] = train_embeddings
        #
        #     path = join(self.precomputed_root, self.test_embedding_fname)
        #     print(f'> Loading test embeddings from {path}...')
        #     with open(path, 'rb') as f:
        #         dict_data['test_embeddings'] = np.load(f)
        # except FileNotFoundError:
        #     print(f'>> Embedding path not found. Calculating new embeddings...')
        #     # setup_train_data = torchvision.datasets.CIFAR10(root=self.dataset_root, train=True, download=False, transform=self.setup_transform)
        #     # setup_test_data = torchvision.datasets.CIFAR10(root=self.dataset_root, train=False, download=False, transform=self.setup_test_transform)
        #     setup_train_data = self.train_data
        #     setup_test_data = self.test_data
        #
        #     all_embeddings = get_embeddings(setup_train_data,
        #                                     setup_test_data,
        #                                     num_epochs=5,  # 10,
        #                                     args=args,
        #                                     stopping_threshold=5)
        #
        #     train_embeddings, test_embeddings = all_embeddings
        #     dict_data['train_embeddings'] = train_embeddings
        #     dict_data['test_embeddings'] = test_embeddings
        #
        # print(f"Train embedding shape: {dict_data['train_embeddings'].shape}")
        # print(f"Test embedding shape: {dict_data['test_embeddings'].shape}")
        #
        # if args.num_distributions == 10:
        #     dist_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #     kmeans_labels = np.array([dist_labels[t] for t in dict_data['targets']])
        #     kmeans_labels_test = np.array([dist_labels[t] for t in dict_data['test_targets']])
        # else:
        #     # 如果是随机分布，即iid分布
        #     # if self.random_dists:
        #     #     print('> Random dataset distribution initialization...')
        #     #     kmeans_labels = np.random.randint(self.num_distributions, size=len(dict_data['inputs']))
        #     #     kmeans_labels_test = np.random.randint(self.num_distributions, size=len(dict_data['test_inputs']))
        #     #
        #     # else:
        #     try:  # First see if kmeans labels were already computed, and load those
        #         path = join(self.precomputed_root, self.kmeans_train_fname)
        #         with open(path, 'rb') as f:
        #             kmeans_labels = np.load(path)
        #         path = join(self.precomputed_root, self.kmeans_test_fname)
        #         print(f'> Loaded clustered labels from {path}!')
        #         with open(path, 'rb') as f:
        #             kmeans_labels_test = np.load(path)
        #         print(f'> Loaded clustered labels from {path}!')
        #     except FileNotFoundError:
        #         # Compute PCA first on combined train and test embeddings
        #         embeddings = np.concatenate([dict_data['train_embeddings'],
        #                                      dict_data['test_embeddings']])
        #         np.random.seed(args.data_seed)
        #         num_components = 256
        #         pca = PCA(n_components=num_components)
        #         dict_data['embeddings'] = pca.fit_transform(embeddings)
        #         print_debug(pca.explained_variance_ratio_.cumsum()[-1],
        #                     f'Ratio of embedding variance explained by {num_components} principal components')
        #
        #         # Compute clusters
        #         np.random.seed(args.data_seed)
        #         km = KMeans(n_clusters=self.args.num_distributions,
        #                     init='k-means++', max_iter=100, n_init=5)
        #
        #         # # For random distributions, specify km.labels_ randomly across the number of distributions
        #         # if self.random_dists:
        #         #     km.labels_ = np.random.randint(self.num_distributions,
        #         #                                    size=len(embeddings))
        #         #     print_debug(len(dict_data['inputs']), 'Size of dataset')
        #         # else:
        #         #     km.fit(dict_data['embeddings'])
        #         km.fit(dict_data['embeddings'])
        #
        #         kmeans_labels_train_test = km.labels_
        #
        #         # Partition into train and test
        #         kmeans_labels = kmeans_labels_train_test[:len(self.train_data.targets)]
        #         kmeans_labels_test = kmeans_labels_train_test[len(self.train_data.targets):]
        #
        #         assert len(kmeans_labels) == len(dict_data['inputs'])
        #         print_debug(len(kmeans_labels_test), 'len(kmeans_labels_test)')
        #         print_debug(len(dict_data['test_inputs']), "len(dict_data['test_inputs'])")
        #         assert len(kmeans_labels_test) == len(dict_data['test_inputs'])
        #
        #         path = join(self.precomputed_root, self.kmeans_train_fname)
        #         with open(path, 'wb') as f:
        #             np.save(f, kmeans_labels)
        #         print(f'> Saved clustered labels to {path}!')
        #
        #         path = join(self.precomputed_root, self.kmeans_test_fname)
        #         with open(path, 'wb') as f:
        #             np.save(f, kmeans_labels_test)
        #         print(f'> Saved clustered labels to {path}!')
        #
        # loaded_images = dict_data['inputs']
        # loaded_labels = dict_data['targets']
        #
        # loaded_images_test = dict_data['test_inputs']
        # loaded_labels_test = dict_data['test_targets']
        #
        # distributions = []
        # distributions_fname = 'distributions.npy'
        # try:
        #     print("load precomputed distribution from disk")
        #     distributions = np.load(f"{self.precomputed_root}/{distributions_fname}")
        # except FileNotFoundError:
        #     # 将聚类好的每个distribution的数据载入字典images_dist，labels_dist，放入distributions[i]
        #     print("no precomputed file was found, newly compute distribution")
        #     for cluster_label in range(self.args.num_distributions):
        #         indices = np.where(kmeans_labels == cluster_label)[0]
        #         images_dist = loaded_images[indices]
        #         labels_dist = loaded_labels[indices]
        #
        #         if self.shuffle:
        #             np.random.seed(args.data_seed)
        #             shuffle_ix = list(range(images_dist.shape[0]))
        #             np.random.shuffle(shuffle_ix)
        #             images_dist = images_dist[shuffle_ix]
        #             labels_dist = labels_dist[shuffle_ix]
        #             indices = indices[shuffle_ix]
        #
        #         test_indices = np.where(kmeans_labels_test == cluster_label)[0]
        #         test_images_dist = loaded_images_test[test_indices]
        #         test_labels_dist = loaded_labels_test[test_indices]
        #
        #         if self.shuffle:
        #             np.random.seed(args.data_seed)
        #             shuffle_ix = list(range(test_images_dist.shape[0]))
        #             np.random.shuffle(shuffle_ix)
        #             test_images_dist = test_images_dist[shuffle_ix]
        #             test_labels_dist = test_labels_dist[shuffle_ix]
        #             test_indices = test_indices[shuffle_ix]
        #
        #         test_loader = DataLoader(torch.utils.data.Subset(self.test_data, indices=test_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        #
        #         # Should be good if the embeddings were calculated in order
        #         distributions.append({'images': images_dist,
        #                               'labels': labels_dist,
        #                               # 'clients': [],
        #                               'clients_data': [],
        #                               'id': cluster_label,
        #                               'indices': indices,
        #                               'test_loader': test_loader,
        #                               'test_labels': test_labels_dist,
        #                               'test_indices': test_indices})
        #     np.save(f"{self.precomputed_root}/{distributions_fname}", distributions)
        # self.init_clients_data(distributions)
        # self.distributions = distributions

        self.distributions = []
        self.init_distributions(True)
        distributions = self.distributions

        self.init_clients_data(self.distributions)
        self.distributions = distributions

    def init_distributions(self, shuffle):
        """
        Initialize client data distributions with through latent non-IID method
        - Groups datapoints into D groupings based on clustering their
          hidden-layer representations from a pre-trained model
        """
        args = self.args
        self.precomputed_root = "./../data/precomputed"
        self.embedding_fname = 'embeddings-d=cifar10-e=4-st=5-lr=0.001-mo=0.9-wd=0.0001-fc2-train.npy'
        self.train_embedding_fname = self.embedding_fname
        self.test_embedding_fname = 'embeddings-d=cifar10-e=4-st=5-lr=0.001-mo=0.9-wd=0.0001-fc2-test.npy'
        kmeans_labels_prefix = f'kmeans-nd={args.num_distributions}-s={args.seed}-ds={args.data_seed}'
        self.kmeans_train_fname = f'{kmeans_labels_prefix}-{self.embedding_fname}'
        self.kmeans_test_fname = f'{kmeans_labels_prefix}-{self.test_embedding_fname}'

        dict_data = {'inputs': np.array(self.train_data.data),
                     'targets': np.array(self.train_data.targets),
                     'test_inputs': np.array(self.test_data.data),
                     'test_targets': np.array(self.test_data.targets)}

        try:  # First try to load pre-computed data
            path = join(self.precomputed_root, self.train_embedding_fname)
            print(f'> Loading training embeddings from {path}...')
            with open(path, 'rb') as f:
                train_embeddings = np.load(f)
                dict_data['train_embeddings'] = train_embeddings

            path = join(self.precomputed_root, self.test_embedding_fname)
            print(f'> Loading test embeddings from {path}...')
            with open(path, 'rb') as f:
                dict_data['test_embeddings'] = np.load(f)

        except FileNotFoundError:
            print(f'>> Embedding path not found. Calculating new embeddings...')
            # setup_train_data = self.train_data
            # setup_test_data = self.test_data

            setup_train_data = torchvision.datasets.CIFAR10(root="./../data",
                                                            train=True,
                                                            download=True,
                                                            transform=self.setup_transform)
            setup_test_data = torchvision.datasets.CIFAR10(root="./../data",
                                                           train=False,
                                                           download=True,
                                                           transform=self.setup_test_transform)

            all_embeddings = get_embeddings(setup_train_data,
                                            setup_test_data,
                                            num_epochs=5,  # 10,
                                            args=args,
                                            stopping_threshold=5)

            train_embeddings, test_embeddings = all_embeddings
            dict_data['train_embeddings'] = train_embeddings
            dict_data['test_embeddings'] = test_embeddings

        print(f"Train embedding shape: {dict_data['train_embeddings'].shape}")
        print(f"Test embedding shape: {dict_data['test_embeddings'].shape}")

        # 聚类
        if args.num_distributions == 10:
            dist_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            kmeans_labels = np.array([dist_labels[t] for t in dict_data['targets']])
            kmeans_labels_test = np.array([dist_labels[t] for t in dict_data['test_targets']])
        else:
            try:  # First see if kmeans labels were already computed, and load those
                path = join(self.precomputed_root, self.kmeans_train_fname)
                with open(path, 'rb') as f:
                    kmeans_labels = np.load(path)
                path = join(self.precomputed_root, self.kmeans_test_fname)
                print(f'> Loaded clustered labels from {path}!')
                with open(path, 'rb') as f:
                    kmeans_labels_test = np.load(path)
                print(f'> Loaded clustered labels from {path}!')
            except FileNotFoundError:
                # Compute PCA first on combined train and test embeddings
                embeddings = np.concatenate([dict_data['train_embeddings'],
                                             dict_data['test_embeddings']])
                np.random.seed(args.data_seed)
                num_components = 256
                pca = PCA(n_components=num_components)
                dict_data['embeddings'] = pca.fit_transform(embeddings)
                print_debug(pca.explained_variance_ratio_.cumsum()[-1],
                            f'Ratio of embedding variance explained by {num_components} principal components')

                # Compute clusters
                np.random.seed(args.data_seed)
                km = KMeans(n_clusters=self.args.num_distributions,
                            init='k-means++', max_iter=100, n_init=5)

                # todo ???不是else吗
                # For random distributions, specify km.labels_ randomly across the number of distributions
                # if self.random_dists:
                #     km.labels_ = np.random.randint(self.args.num_distributions,
                #                                    size=len(embeddings))
                #     print_debug(len(dict_data['inputs']), 'Size of dataset')
                # else:
                #     km.fit(dict_data['embeddings'])
                km.fit(dict_data['embeddings'])

                kmeans_labels_train_test = km.labels_
                # Partition into train and test
                kmeans_labels = kmeans_labels_train_test[:len(self.train_data.targets)]
                kmeans_labels_test = kmeans_labels_train_test[len(self.train_data.targets):]

                assert len(kmeans_labels) == len(dict_data['inputs'])
                print_debug(len(kmeans_labels_test), 'len(kmeans_labels_test)')
                print_debug(len(dict_data['test_inputs']), "len(dict_data['test_inputs'])")
                assert len(kmeans_labels_test) == len(dict_data['test_inputs'])

                path = join(self.precomputed_root, self.kmeans_train_fname)
                with open(path, 'wb') as f:
                    np.save(f, kmeans_labels)
                print(f'> Saved clustered labels to {path}!')

                path = join(self.precomputed_root, self.kmeans_test_fname)
                with open(path, 'wb') as f:
                    np.save(f, kmeans_labels_test)
                print(f'> Saved clustered labels to {path}!')

        loaded_images = dict_data['inputs']
        loaded_labels = dict_data['targets']

        loaded_images_test = dict_data['test_inputs']
        loaded_labels_test = dict_data['test_targets']

        # 将聚类好的每个distribution的数据载入字典images_dist，labels_dist
        for cluster_label in range(self.args.num_distributions):
            indices = np.where(kmeans_labels == cluster_label)[0]
            images_dist = loaded_images[indices]
            labels_dist = loaded_labels[indices]

            if shuffle:
                np.random.seed(args.data_seed)
                shuffle_ix = list(range(images_dist.shape[0]))
                np.random.shuffle(shuffle_ix)
                images_dist = images_dist[shuffle_ix]
                labels_dist = labels_dist[shuffle_ix]
                indices = indices[shuffle_ix]

            test_indices = np.where(kmeans_labels_test == cluster_label)[0]
            test_images_dist = loaded_images_test[test_indices]
            test_labels_dist = loaded_labels_test[test_indices]

            if shuffle:
                np.random.seed(args.data_seed)
                shuffle_ix = list(range(test_images_dist.shape[0]))
                np.random.shuffle(shuffle_ix)
                test_images_dist = test_images_dist[shuffle_ix]
                test_labels_dist = test_labels_dist[shuffle_ix]
                test_indices = test_indices[shuffle_ix]

            # Should be good if the embeddings were calculated in order
            self.distributions.append({'images': images_dist,
                                       'labels': labels_dist,
                                       'clients': [],
                                       'id': cluster_label,
                                       'clients_data': [],
                                       'indices': indices,
                                       'test_labels': test_labels_dist,
                                       'test_indices': test_indices})

        # 打印划分结果
        for d in range(len(self.distributions)):
            dist_dict = self.distributions[d]
            labels = dist_dict['labels']
            labels, counts = np.unique(labels, return_counts=True)
            counts = ' '.join([f'({labels[i]:2d}, {counts[i]:4d})' for i in range(len(labels))])
            print(f'Distribution {d} train (labels, counts)')
            print(f'{counts}')
            labels = dist_dict['test_labels']
            labels, counts = np.unique(labels, return_counts=True)
            counts = ' '.join([f'({labels[i]:2d}, {counts[i]:4d})' for i in range(len(labels))])
            print(f'Distribution {d} test (labels, counts)')
            print(f'{counts}')

    def init_clients_data(self, distributions):
        args = self.args
        print('> Initializing clients...')

        print(f'>> Total clients: {self.args.client_num_in_total}')
        print(f'>> Clients per distribution: {self.args.num_clients_per_dist}')

        self.test_non_iid = [None for _ in range(len(distributions))]

        # 将每个distribution里的数据划分成num_clients_per_dist份，以供后续client初始化
        for ix, dist in enumerate(distributions):
            # Already shuffled during initialization, but can do again if desired
            np.random.seed(self.args.data_seed)
            shuffle_ix = list(range(len(distributions[ix]['images'])))
            np.random.shuffle(shuffle_ix)

            images = dist['images'][shuffle_ix]
            labels = dist['labels'][shuffle_ix]
            indices = dist['indices'][shuffle_ix]

            # todo why???
            # Transpose images and numpy dims
            # if dist['images'].shape[1] != 3:
            #     dims = dist['images'][shuffle_ix].shape
            #
            #     images = images.transpose([0, 3, 1, 2])
            #     dims_ = images.shape
            #     print(f'Transposing image dims for data loading: {dims} => {dims_}')

            # 为每个client分配数据
            data_by_clients = np.array_split(images, self.args.num_clients_per_dist)
            labels_by_clients = np.array_split(labels, self.args.num_clients_per_dist)

            indices_by_clients = np.array_split(indices, self.args.num_clients_per_dist)

            # Do the same for test set
            test_indices = dist['test_indices']
            test_indices_by_clients = np.array_split(test_indices, self.args.num_clients_per_dist)

            non_iid_test_set = torch.utils.data.Subset(self.test_data, indices=test_indices)
            self.test_non_iid[ix] = DataLoader(non_iid_test_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

            # 分配distribution中的数据给client
            for i in range(self.args.num_clients_per_dist):

                # 确保data和labels能够对上
                assert data_by_clients[i].shape[0] == labels_by_clients[i].shape[0]

                client_indices = indices_by_clients[i]
                client_test_indices = test_indices_by_clients[i]

                client_dataset = torch.utils.data.Subset(self.train_data, indices=client_indices)
                client_test_dataset = torch.utils.data.Subset(self.test_data, indices=client_test_indices)

                client_targets = [self.train_data.targets[x] for x in client_indices]
                client_unique_classes = np.unique(client_targets)

                # 划分validation
                train_split_size = int(np.round(self.args.train_split * len(client_dataset)))
                val_split_size = len(client_dataset) - train_split_size

                client_train_dataset, client_val_dataset = torch.utils.data.random_split(client_dataset,
                                                                                         [train_split_size,
                                                                                          val_split_size])
                dist['clients_data'].append([ix, client_indices, client_test_indices, client_train_dataset,
                                             client_val_dataset, client_test_dataset, client_unique_classes])

        # if args.parallelize:
        #     pass
        # else:
        #     # Compute EMD
        #     average_emd = np.mean([compute_emd(client.targets, self.train_data.targets) for client in self.clients])
        #     print_header(f'> Global mean EMD: {average_emd}')
        #     for client in self.clients:
        #         client.EMD = average_emd
        #     if args.num_distributions > 1:
        #         self.finalize_client_datasets()
        self.dist_client_dict = [[] for _ in range(len(distributions))]
        c_id = 0
        for ix in range(len(distributions)):
            dist = distributions[ix]
            for i in range(args.num_clients_per_dist):
                client_data = dist['clients_data'][i]

                self.train_loader.append(
                    DataLoader(client_data[3], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers))
                self.validation_loader.append(
                    DataLoader(client_data[4], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers))
                self.test_loader.append(
                    DataLoader(client_data[5], batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers))
                self.dist_client_dict[client_data[0]].append(c_id)
                self.client_dist_dict[c_id] = client_data[0]
                self.train_idx_dict[c_id] = client_data[1]
                c_id += 1

        for ix in range(len(distributions)):
            print(f"分布{ix}具有client:[{self.dist_client_dict[ix]}]")

        # 计算emd，评估
        emd_list = []
        for ix, dist in enumerate(distributions):
            for i in range(self.args.num_clients_per_dist):
                # print(dist['clients_data'][i][1])
                emd_list.append(compute_emd([self.train_data.targets[x] for x in dist['clients_data'][i][1]], self.train_data.targets))
        # average_emd = np.mean(emd_list, axis=0)
        print(emd_list)
        average_emd = np.mean(emd_list)
        print(f'> Global mean EMD : {average_emd}')

        # print_debug([f'Dist: {c.dist_id}, id: {c.id}, adversarial: {c.adversarial}' for c in self.clients])

    # def init_distributions(self, shuffle):
    #     """
    #     Initialize client data distributions with through latent non-IID method
    #     - Groups datapoints into D groupings based on clustering their
    #       hidden-layer representations from a pre-trained model
    #     """
    #     dict_data = {'inputs': np.array(self.train_data.data),
    #                  'targets': np.array(self.train_data.targets),
    #                  'test_inputs': np.array(self.test_data.data),
    #                  'test_targets': np.array(self.test_data.targets)}
    #
    #     try:  # First try to load pre-computed data
    #         path = join(self.precomputed_root, self.train_embedding_fname)
    #         print(f'> Loading training embeddings from {path}...')
    #         with open(path, 'rb') as f:
    #             train_embeddings = np.load(f)
    #             dict_data['train_embeddings'] = train_embeddings
    #
    #         path = join(self.precomputed_root, self.test_embedding_fname)
    #         print(f'> Loading test embeddings from {path}...')
    #         with open(path, 'rb') as f:
    #             dict_data['test_embeddings'] = np.load(f)
    #
    #     except FileNotFoundError:
    #         print(f'>> Embedding path not found. Calculating new embeddings...')
    #         setup_train_data = torchvision.datasets.CIFAR10(root=self.dataset_root,
    #                                                         train=True,
    #                                                         download=False,
    #                                                         transform=self.setup_transform)
    #         setup_test_data = torchvision.datasets.CIFAR10(root=self.dataset_root,
    #                                                        train=False,
    #                                                        download=False,
    #                                                        transform=self.setup_test_transform)
    #         all_embeddings = get_embeddings(setup_train_data,
    #                                         setup_test_data,
    #                                         num_epochs=5,  # 10,
    #                                         args=args,
    #                                         stopping_threshold=5)
    #
    #         train_embeddings, test_embeddings = all_embeddings
    #         dict_data['train_embeddings'] = train_embeddings
    #         dict_data['test_embeddings'] = test_embeddings
    #
    #     print(f"Train embedding shape: {dict_data['train_embeddings'].shape}")
    #     print(f"Test embedding shape: {dict_data['test_embeddings'].shape}")
    #
    #     if args.num_distributions == 10:
    #         dist_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #         kmeans_labels = np.array([dist_labels[t] for t in dict_data['targets']])
    #         kmeans_labels_test = np.array([dist_labels[t] for t in dict_data['test_targets']])
    #     else:
    #         # 如果是随机分布，即iid分布
    #         if self.random_dists:
    #             print('> Random dataset distribution initialization...')
    #             kmeans_labels = np.random.randint(self.num_distributions, size=len(dict_data['inputs']))
    #             kmeans_labels_test = np.random.randint(self.num_distributions, size=len(dict_data['test_inputs']))
    #
    #         else:
    #             try:  # First see if kmeans labels were already computed, and load those
    #                 path = join(self.precomputed_root, self.kmeans_train_fname)
    #                 with open(path, 'rb') as f:
    #                     kmeans_labels = np.load(path)
    #                 path = join(self.precomputed_root, self.kmeans_test_fname)
    #                 print(f'> Loaded clustered labels from {path}!')
    #                 with open(path, 'rb') as f:
    #                     kmeans_labels_test = np.load(path)
    #                 print(f'> Loaded clustered labels from {path}!')
    #             except FileNotFoundError:
    #                 # Compute PCA first on combined train and test embeddings
    #                 embeddings = np.concatenate([dict_data['train_embeddings'],
    #                                              dict_data['test_embeddings']])
    #                 np.random.seed(args.data_seed)
    #                 num_components = 256
    #                 pca = PCA(n_components=num_components)
    #                 dict_data['embeddings'] = pca.fit_transform(embeddings)
    #                 print_debug(pca.explained_variance_ratio_.cumsum()[-1],
    #                             f'Ratio of embedding variance explained by {num_components} principal components')
    #
    #                 # Compute clusters
    #                 np.random.seed(args.data_seed)
    #                 km = KMeans(n_clusters=self.num_distributions,
    #                             init='k-means++', max_iter=100, n_init=5)
    #
    #                 # todo ???不是else吗
    #                 # For random distributions, specify km.labels_ randomly across the number of distributions
    #                 if self.random_dists:
    #                     km.labels_ = np.random.randint(self.num_distributions,
    #                                                    size=len(embeddings))
    #                     print_debug(len(dict_data['inputs']), 'Size of dataset')
    #                 else:
    #                     km.fit(dict_data['embeddings'])
    #
    #                 kmeans_labels_train_test = km.labels_
    #                 # Partition into train and test
    #                 kmeans_labels = kmeans_labels_train_test[:len(self.train_data.targets)]
    #                 kmeans_labels_test = kmeans_labels_train_test[len(self.train_data.targets):]
    #
    #                 assert len(kmeans_labels) == len(dict_data['inputs'])
    #                 print_debug(len(kmeans_labels_test), 'len(kmeans_labels_test)')
    #                 print_debug(len(dict_data['test_inputs']), "len(dict_data['test_inputs'])")
    #                 assert len(kmeans_labels_test) == len(dict_data['test_inputs'])
    #
    #                 path = join(self.precomputed_root, self.kmeans_train_fname)
    #                 with open(path, 'wb') as f:
    #                     np.save(f, kmeans_labels)
    #                 print(f'> Saved clustered labels to {path}!')
    #
    #                 path = join(self.precomputed_root, self.kmeans_test_fname)
    #                 with open(path, 'wb') as f:
    #                     np.save(f, kmeans_labels_test)
    #                 print(f'> Saved clustered labels to {path}!')
    #
    #     loaded_images = dict_data['inputs']
    #     loaded_labels = dict_data['targets']
    #
    #     loaded_images_test = dict_data['test_inputs']
    #     loaded_labels_test = dict_data['test_targets']
    #
    #     # 将聚类好的每个distribution的数据载入字典images_dist，labels_dist
    #     for cluster_label in range(self.num_distributions):
    #         indices = np.where(kmeans_labels == cluster_label)[0]
    #         images_dist = loaded_images[indices]
    #         labels_dist = loaded_labels[indices]
    #
    #         if shuffle:
    #             np.random.seed(args.data_seed)
    #             shuffle_ix = list(range(images_dist.shape[0]))
    #             np.random.shuffle(shuffle_ix)
    #             images_dist = images_dist[shuffle_ix]
    #             labels_dist = labels_dist[shuffle_ix]
    #             indices = indices[shuffle_ix]
    #
    #         test_indices = np.where(kmeans_labels_test == cluster_label)[0]
    #         test_images_dist = loaded_images_test[test_indices]
    #         test_labels_dist = loaded_labels_test[test_indices]
    #
    #         if shuffle:
    #             np.random.seed(args.data_seed)
    #             shuffle_ix = list(range(test_images_dist.shape[0]))
    #             np.random.shuffle(shuffle_ix)
    #             test_images_dist = test_images_dist[shuffle_ix]
    #             test_labels_dist = test_labels_dist[shuffle_ix]
    #             test_indices = test_indices[shuffle_ix]
    #
    #         # Should be good if the embeddings were calculated in order
    #         self.distributions.append({'images': images_dist,
    #                                    'labels': labels_dist,
    #                                    'clients': [],
    #                                    'id': cluster_label,
    #                                    'indices': indices,
    #                                    'test_labels': test_labels_dist,
    #                                    'test_indices': test_indices})
    #
    #     # 打印划分结果
    #     for d in range(len(self.distributions)):
    #         dist_dict = self.distributions[d]
    #         labels = dist_dict['labels']
    #         labels, counts = np.unique(labels, return_counts=True)
    #         counts = ' '.join([f'({labels[i]:2d}, {counts[i]:4d})' for i in range(len(labels))])
    #         print(f'Distribution {d} train (labels, counts)')
    #         print(f'{counts}')
    #         labels = dist_dict['test_labels']
    #         labels, counts = np.unique(labels, return_counts=True)
    #         counts = ' '.join([f'({labels[i]:2d}, {counts[i]:4d})' for i in range(len(labels))])
    #         print(f'Distribution {d} test (labels, counts)')
    #         print(f'{counts}')
    #
    # def init_clients(self):
    #     # print_header('> Initializing clients...')
    #
    #     print(f'>> Total clients: {self.num_clients}')
    #     print(f'>> Clients per distribution: {self.num_clients_per_dist}')
    #
    #     ix = 0
    #     dist_ix = 0
    #
    #     # Fill in clients
    #     if cfg.CLIENT.MANUAL:
    #         for client_params in cfg.CLIENT.POPULATION:
    #             client = Client(client_id=client_params['client_id'])
    #             client.local_val_ratio = client_params['lvr']
    #             client.shared_val = client_params['shared_val']
    #             self.distributions[client_params['dist_id']]['clients'].append(client)
    #     else:
    #         while ix < self.num_clients and dist_ix < len(self.distributions):
    #             client = Client(client_id=ix)
    #             client.local_val_ratio = args.local_val_ratio
    #             client.shared_val = True
    #             self.distributions[dist_ix]['clients'].append(client)
    #             if (ix + 1) % self.num_clients_per_dist == 0:
    #                 dist_ix += 1
    #             ix += 1
    #         for i in range(ix, self.num_clients):
    #             client = Client(client_id=i)
    #             client.local_val_ratio = args.local_val_ratio
    #             client.shared_val = True
    #             print_debug(i % self.num_distributions, 'client mod')
    #             self.distributions[i % self.num_distributions]['clients'].append(client)
    #
    #     tqdm_clients = tqdm(total=self.num_clients)
    #     np.random.seed(args.data_seed)
    #     for ix, dist in enumerate(self.distributions):
    #         # Already shuffled during initialization, but can do again if desired
    #         np.random.seed(args.data_seed)
    #         shuffle_ix = list(range(len(self.distributions[ix]['images'])))
    #         np.random.shuffle(shuffle_ix)
    #
    #         images = dist['images'][shuffle_ix]
    #         labels = dist['labels'][shuffle_ix]
    #         indices = dist['indices'][shuffle_ix]
    #
    #         # Transpose images and numpy dims
    #         if dist['images'].shape[1] != 3:
    #             dims = dist['images'][shuffle_ix].shape
    #             images = images.transpose([0, 3, 1, 2])
    #             dims_ = images.shape
    #             print(f'Transposing image dims for data loading: {dims} => {dims_}')
    #
    #         # 为每个client分配数据
    #         data_by_clients = np.array_split(images, len(dist['clients']))
    #         labels_by_clients = np.array_split(labels, len(dist['clients']))
    #         indices_by_clients = np.array_split(indices, len(dist['clients']))
    #
    #         # Do the same for test set
    #         test_indices = dist['test_indices']
    #         test_indices_by_clients = np.array_split(test_indices, len(dist['clients']))
    #
    #         # Initialize clients
    #         for cix, client in enumerate((dist['clients'])):
    #             client.population = self  # Give access to the population for each client
    #
    #             # Setup data
    #             assert data_by_clients[cix].shape[0] == labels_by_clients[cix].shape[0]
    #
    #             # Setup total local dataset for each client
    #             client_dataset = torch.utils.data.Subset(self.train_data, indices=indices_by_clients[cix])
    #             client_test_dataset = torch.utils.data.Subset(self.test_data, indices=test_indices_by_clients[cix])
    #
    #             client.initialize(client_dataset, dist, client_test_dataset)
    #             client.targets = [self.train_data.targets[x] for x in indices_by_clients[cix]]
    #             client.unique_classes = np.unique(client.targets)
    #             print(f"client {client.id} unique class: {client.unique_classes}")
    #
    #             self.clients[client.id] = client  # Give another reference to the client
    #             tqdm_clients.update(n=1)
    #
    #             client.dataset_train_indices = indices_by_clients[cix]
    #             train_split_size = int(np.round(len(indices_by_clients[cix]) * args.train_split))
    #             # Set up client indices specifically for local training set
    #             client.local_train_indices = np.random.choice(client.dataset_train_indices,
    #                                                           size=train_split_size, replace=False)
    #             client.dataset_test_indices = test_indices_by_clients[cix]
    #
    #     tqdm_clients.close()
    #
    #     if args.parallelize:
    #         pass
    #     else:
    #         # Compute EMD
    #         average_emd = np.mean([compute_emd(client.targets, self.train_data.targets) for client in self.clients])
    #
    #         print_header(f'> Global mean EMD: {average_emd}')
    #         for client in self.clients:
    #             client.EMD = average_emd
    #         if args.num_distributions > 1:
    #             self.finalize_client_datasets()
    #
    #     print_debug([f'Dist: {c.dist_id}, id: {c.id}, adversarial: {c.adversarial}' for c in self.clients])