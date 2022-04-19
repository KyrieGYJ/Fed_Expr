import logging

import numpy as np
import wandb

from FedML.fedml_api.standalone.decentralized.client_pushsum import ClientPushsum
from FedML.fedml_api.standalone.decentralized.topology_manager import TopologyManager
from MyExpr.ml.SUSY2 import ClientTEST


# 主训练方法
def myexpr_decentralized_fl(client_number, client_id_list, streaming_data, model, model_cache, args):
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

            client = ClientTEST(model, model_cache, client_id, client_data, topology_manager,
                                iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric)

        else:
            client = ClientTEST(model, model_cache, client_id, client_data, topology_manager,
                                iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric)

        client_list.append(client)

    log_file_path = "./log/decentralized_fl.txt"
    f_log = open(log_file_path, mode='w+', encoding='utf-8')

    # training
    for t in range(iteration_number_T * epoch):
        logging.info('--- Iteration %d ---' % t)

        for client in client_list:
            # todo 插入广播策略（在随机广播的基础上）
            # 修改顺序：在训练前就把模型发送出去，以便联合更新
            client.train(t)
            client.send_local_gradient_to_neighbor(client_list)

        # line 6: update
        for client in client_list:
            #  用topK方法选出有效的k个邻居
            #  todo 根据f1
            top_k = client.top_k_by_loss(client_list, t)

            # 把联合更新策略升级为深度互学习
            client.mutual_update(top_k)

        regret = cal_regret(client_list, client_number, t)
        # print("regret = %s" % regret)
        loss = cal_average_loss(client_list, client_number, t)
        acc = cal_acc(client_list, t)

        # todo f1指标
        wandb.log({"Average Loss": regret, "loss": loss, "acc": acc, "iteration": t})

        f_log.write("%f,%f\n" % (t, regret))

    f_log.close()
    wandb.save(log_file_path)


# Average Loss (global)
def cal_regret(client_list, client_number, t):
    regret = 0
    for client in client_list:
        regret += np.sum(client.get_regret())

    # 总共t+1轮训练，取 regret = 总loss / （客户数量 * 总轮次）
    regret = regret / (client_number * (t + 1))
    return regret


# Average Loss (single round)(有问题，应该去掉)
def cal_average_loss(client_list, client_number, t):
    loss = 0
    for client in client_list:
        loss += client.get_regret()[t]
    loss = loss / client_number
    return loss


# acc
def cal_acc(client_list, t):
    correct = 0
    total = 0
    for client in client_list:
        record = client.get_record()
        correct += record[t][0]
        total += record[t][1]

    return correct / total
