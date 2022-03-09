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

    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    # client相关

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--topology_neighbors_num_undirected', type=int, default=4)

    parser.add_argument('--topology_neighbors_num_directed', type=int, default=4)


    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0, help='CI')

    parser.add_argument('--topK_strategy', type=str, default="loss", help='[loss, f1]')

    parser.add_argument('--broadcaster_strategy', type=str, default="flood", help='[flood, random, cluster, similarity]')

    parser.add_argument('--mutual_trainer_strategy', type=str, default="local_and_mutual", help='[local_and_mutual, mutual, pushsum, dsgd]')

    parser.add_argument('--num_workers', type=int, default=2)

    return parser