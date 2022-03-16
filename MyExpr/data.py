import torch
import os.path
from torchvision.datasets import utils, MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from PIL import Image
import numpy as np
from MyExpr.utils import compute_emd


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
        self.train_all, self.test_all, self.train_loader, self.test_loader = None, None, None, None
        self.generate_loader = None
        if args.data_distribution == "iid":
            self.generate_loader = self.iid
        elif args.data_distribution == "non-iid":
            self.generate_loader = self.non_iid

    # todo 改编自LG-FedAvg
    def non_iid(self):
        train_data = self.train_data
        test_data = self.test_data
        args = self.args

        # dict_users_train 记录了每个用户持有的数据下标集合
        dict_users_train, rand_set_all = self.client_noniid(train_data, args.client_num_in_total, args.shards_per_user,
                                                       rand_set_all=[], args=args)
        dict_users_test, rand_set_all = self.client_noniid(test_data, args.client_num_in_total, args.shards_per_user,
                                                      rand_set_all=rand_set_all, args=args)

        client_train_dataset = []
        client_validation_dataset = []
        for ix in dict_users_train:
            client_dataset = torch.utils.data.Subset(train_data, indices=dict_users_train[ix])
            print("client[{}] 包含的类为:{}".format(ix, np.unique(torch.tensor(train_data.targets)[dict_users_train[ix]])))
            # 划分validation
            len_train_split = int(np.round(args.train_split * len(client_dataset)))
            len_val_split = len(client_dataset) - len_train_split

            torch.manual_seed(args.data_seed)
            split_lens = [len_train_split, len_val_split]
            # print(split_lens)
            # print(len(dataset))
            datasets = torch.utils.data.random_split(client_dataset, split_lens)
            client_train_dataset.append(datasets[0])
            client_validation_dataset.append(datasets[1])

        # client_dataset = [torch.utils.data.Subset(train_data, indices=dict_users_train[ix]) for ix in dict_users_train]
        client_test_dataset = [torch.utils.data.Subset(test_data, indices=dict_users_test[ix]) for ix in dict_users_train]


        self.train_loader = [DataLoader(client_train_dataset[i], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                             for i in dict_users_train]

        self.validation_loader = [DataLoader(client_validation_dataset[i], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                             for i in dict_users_train]

        self.test_loader = [DataLoader(client_test_dataset[i], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                             for i in dict_users_train]

        self.train_all = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        self.test_all = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # emd = []
        # for ix in dict_users_train:
        #     client_targets = [train_data.targets[x] for x in dict_users_train[ix]]
        #     emd.append(compute_emd(client_targets, train_data.targets))

        # average_emd = np.mean([compute_emd(client.targets, train_data.targets) for client in self.clients])
        average_emd = np.mean([compute_emd([train_data.targets[x] for x in dict_users_train[ix]], train_data.targets) for ix in dict_users_train], axis=0)
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

        self.train_loader = [DataLoader(splited_trainset[i], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                             for i in range(args.client_num_in_total)]

        self.test_loader = [DataLoader(splited_testset[i], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                            for i in range(args.client_num_in_total)]
        # self.test_loader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    def client_noniid(self, dataset, num_users, shard_per_user, rand_set_all, args):
        """
        Sample non-IID client data from dataset in pathological manner - from LG-FedAvg implementation
        :param dataset:
        :param num_users:
        :return: (dictionary, where keys = client_id / index, and values are dataset indices), rand_set_all (all classes)

        shard_per_user should be a factor of the dataset size
        """
        seed = args.data_seed
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

        idxs_dict = {}

        # 将dataset中的数据（下标）放入字典
        for i in range(len(dataset)):
            label = torch.tensor(dataset.targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

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
