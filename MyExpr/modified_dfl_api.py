import heapq
import logging
import math

import numpy as np
import wandb
import torch

from FedML.fedml_api.standalone.decentralized.client_dsgd import ClientDSGD
from FedML.fedml_api.standalone.decentralized.client_pushsum import ClientPushsum
from FedML.fedml_api.standalone.decentralized.topology_manager import TopologyManager


def cal_regret(client_list, client_number, t):
    regret = 0
    for client in client_list:
        regret += np.sum(client.get_regret())

    # 总共t+1轮训练，取 regret = 总loss / （客户数量 * 总轮次）
    regret = regret / (client_number * (t + 1))
    return regret


# 主训练方法
def FedML_decentralized_fl(client_number, client_id_list, streaming_data, model, model_cache, args):
    # 读参数
    iteration_number_T = args.iteration_number
    lr_rate = args.learning_rate
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    topology_neighbors_num_undirected = args.topology_neighbors_num_undirected
    topology_neighbors_num_directed = args.topology_neighbors_num_directed
    latency = args.latency
    b_symmetric = args.b_symmetric
    epoch = args.epoch
    time_varying = args.time_varying

    # create the network topology topology
    logging.info("generating topology")
    if b_symmetric:
        topology_manager = TopologyManager(client_number, True,
                                           undirected_neighbor_num=topology_neighbors_num_undirected)
    else:
        topology_manager = TopologyManager(client_number, False,
                                           undirected_neighbor_num=topology_neighbors_num_undirected,
                                           out_directed_neighbor=topology_neighbors_num_directed)
    topology_manager.generate_topology()
    logging.info("finished topology generation")

    # create all client instances (each client will create an independent model instance)
    client_list = []
    for client_id in client_id_list:
        client_data = streaming_data[client_id]
        # print("len = " + str(len(client_data)))

        if args.mode == 'PUSHSUM':

            client = ClientPushsum(model, model_cache, client_id, client_data, topology_manager,
                                   iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                   weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric,
                                   time_varying=time_varying)

        elif args.mode == 'DOL':

            client = ClientDSGD(model, model_cache, client_id, client_data, topology_manager,
                                iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric)

        else:
            client = ClientDSGD(model, model_cache, client_id, client_data, topology_manager,
                                iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric)

        client_list.append(client)

    log_file_path = "./log/decentralized_fl.txt"
    f_log = open(log_file_path, mode='w+', encoding='utf-8')

    # training
    for t in range(iteration_number_T * epoch):
        logging.info('--- Iteration %d ---' % t)

        # todo 读论文看一下这一块算法
        if args.mode == 'DOL' or args.mode == 'PUSHSUM':
            for client in client_list:

                # line 4: Locally computes the intermedia variable
                client.train(t)

                # todo 插入广播策略（在随机广播的基础上）
                # line 5: send to neighbors
                client.send_local_gradient_to_neighbor(client_list)




            # line 6: update
            for client in client_list:
                # todo 用top方法选出有效的k个邻居 插入根据loss，f1，取top k个有效模型
                top_k = top_k_by_loss(client, client_list, t)

                # todo 把联合更新策略升级为深度互学习
                update_local_parameters_by_mutual_learning(client, top_k)

                # 边权重聚合模型，貌似这样就是模型插值吧
                client.update_local_parameters()
        else:
            for client in client_list:
                client.train_local(t)

        regret = cal_regret(client_list, client_number, t)
        # print("regret = %s" % regret)

        wandb.log({"Average Loss": regret, "iteration": t})

        f_log.write("%f,%f\n" % (t, regret))

    f_log.close()
    wandb.save(log_file_path)


def top_k_by_loss(client, client_list, t):
    heap = []
    train_x = torch.from_numpy(client.streaming_data[t]['x']).float()
    train_y = torch.FloatTensor([client.streaming_data[t]['y']])
    for index in range(len(client.topology)):
        if client.topology[index] != 0 and index != client.id:
            neighbor = client_list[index]
            loss = neighbor.criterion(neighbor.model(train_x), train_y)
            heap.append([index, loss])
    top_k = heapq.nlargest(math.floor(len(heap)*0.8), heap, lambda x : x[1])
    return top_k

def update_local_parameters_by_mutual_learning(client, top_k):
