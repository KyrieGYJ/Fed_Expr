import argparse

def add_args():
    parser = argparse.ArgumentParser(description='DFL-Mutual-standalone')
    # Training settings
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../data',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained locally')

    # client相关

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--topology_neighbors_num_undirected', type=int, default=20)

    parser.add_argument('--topology_neighbors_num_directed', type=int, default=0)

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')

    parser.add_argument('--ci', type=int, default=0, help='CI')

    parser.add_argument('--topK_strategy', type=str, default="loss", help='[loss, f1_marco, f1_micro]')

    parser.add_argument('--broadcaster_strategy', type=str, default="flood",
                        help='[flood, affinity, random, cluster, similarity]')

    parser.add_argument('--trainer_strategy', type=str, default="local_and_mutual",
                        help='[local_and_mutual, mutual, local_train, model_interpolation, pushsum]')

    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--name', type=str, default='',
                        help='name of current run')

    parser.add_argument('--data_distribution', type=str, default='iid',
                        help='iid, non-iid')

    parser.add_argument('--data_seed', type=int, default=0, help="data seed")

    parser.add_argument('--shards_per_user', type=int, default=2, help="2, 3, 4, 5, 10")

    parser.add_argument('--train_split', type=float, default=0.8, help="训练数据集比例")

    parser.add_argument('--communication_wise', type=str, default='epoch', help="iteration, epoch")

    parser.add_argument('--local_train_stop_point', type=int, default=99999999,
                        help="communication round where local train stop")

    return parser