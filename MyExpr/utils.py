import numpy as np
from collections import Counter
import heapq
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import copy
from torch.utils.data import DataLoader

# report the average Earth Mover’s Distance (EMD) between local client data and the total dataset
# across all clients to quantify non-IIDness.
def compute_emd(targets_1, targets_2):
    """Calculates Earth Mover's Distance between two array-like objects (dataset labels)"""
    total_targets = []
    total_targets.extend(list(np.unique(targets_1)))
    total_targets.extend(list(np.unique(targets_2)))

    emd = 0

    counts_1 = Counter(targets_1)
    counts_2 = Counter(targets_2)

    size_1 = len(targets_1)
    size_2 = len(targets_2)

    for t in counts_1:
        count_2 = counts_2[t] if t in counts_2 else 0
        emd += np.abs((counts_1[t] / size_1) - (count_2 / size_2))

    for t in counts_2:
        count_1 = counts_1[t] if t in counts_1 else 0
        emd += np.abs((counts_2[t] / size_2) - (count_1 / size_1))

    return emd


# 计算模型之间的权重(参考ICLR FedFomo)
def cal_w(client, args):
    """
    计算本地模型与其他模型的权重
    ：local_para  本地模型参数
    : parameters 其他模型参数集合
    : return  权重列表
    """
    # 因为在local_update后broadcast，采用本轮local_update后的模型和上一轮接收到的模型进行计算
    # 相当于基于接受者上一轮的模型预测：接受者的本地模型在本地经历过mutual_update和新一轮local_update后，发送者的新本地模型和它的仍然是相近的。
    w = {}
    criterion = client.criterion_CE
    val_dataloader = client.validation_loader
    local_model = client.model
    received_dic = client.last_received_model_dict
    # train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    # Net.load_state_dict(local_model, strict=True)

    total_local_loss = 0.0
    loss_dic = {}
    for c_id, model in received_dic.items():
        loss_dic[c_id] = 0.0

    with torch.no_grad():

        for data, label in val_dataloader:
            data, label = data.to(args.device), label.to(args.device)
            local_outputs = local_model(data)
            local_loss = criterion(local_outputs, label)

            if "cuda" in args.device:
                local_loss = local_loss.cpu()

            total_local_loss += local_loss.item()

            for c_id, model in received_dic.items():
                received_outputs = model(data)
                received_loss = criterion(received_outputs, label)

                if "cuda" in args.device:
                    received_loss = received_loss.cpu()

                loss_dic[c_id] += received_loss.item()
    epsilon = 1e-6
    for c_id, model in received_dic.items():
        # 计算分母
        dif = compute_parameter_difference(model, local_model, norm=args.model_delta_norm)
        # dif = CalDif(local_model, model, args)
        w[c_id] = ((total_local_loss - loss_dic[c_id]) / (dif + epsilon))

    return w


# 搬运FedFomo，功能与CalDif相同
def compute_parameter_difference(model_a, model_b, norm='l2'):
    """
    Compute difference in two model parameters
    """
    if norm == 'l1':
        total_diff = 0.
        total_diff_l2 = 0.
        # Compute L1-norm, i.e. ||w_a - w_b||_1
        for w_a, w_b in zip(model_a.parameters(), model_b.parameters()):
            total_diff += (w_a - w_b).norm(1).item()
            total_diff_l2 += torch.pow((w_a - w_b).norm(2), 2).item()

        return total_diff

    elif norm == 'l2_root':
        total_diff = 0.
        for w_a, w_b in zip(model_a.parameters(), model_b.parameters()):
            total_diff += (w_a - w_b).norm(2).item()
        return total_diff

    total_diff = 0.
    model_a_params = []
    for p in model_a.parameters():
        model_a_params.append(p.detach().cpu().numpy().astype(np.float64))

    for ix, p in enumerate(model_b.parameters()):
        p_np = p.detach().cpu().numpy().astype(np.float64)
        diff = model_a_params[ix] - p_np
        scalar_diff = np.sum(diff ** 2)
        total_diff += scalar_diff
    # Can be vectorized as
    # np.sum(np.power(model_a.parameters().detach().cpu().numpy() -
    #                 model_a.parameters().detach().cpu().numpy(), 2))
    return total_diff  # Returns distance^2 between two model parameters


# 计算模型之间的欧式距离
def CalDif(model1, model2, args):
    # 利用矩阵范数求模型间的差异
    # 比较粗糙的方式
    # 或者在算loss的时候，顺便把准确率算出来，作为模型间差异
    dif = 0
    with torch.no_grad():
        # model.parameters返回的是迭代器，model.state_dict返回的是参数字典
        for item1, item2 in zip(list(model1.parameters()), list(model2.parameters())):
            para_dif = torch.norm(item1 - item2)

            if "cuda" in args.device:
                para_dif = para_dif.cpu()

            dif += para_dif
    return dif


def get_adjacency_matrix(client, symmetric=True, shift_positive=False,
                         normalize=True, rbf_kernel=False, rbf_delta=1.):
    """
    If learning federations through updated client weight preferences,
    first compute adjacency matrix given client-to-client weights
    Args:
    - clients (Clients[]): list of clients <-- should be the population.clients array
    - symmetric (bool): return a symmetric matrix
    - shift_positive (bool): shift all values to be >= 0 (add (0 - lowest value) to everything)
    - normalize (bool): after other transformations, normalize everything to range [0, 1]
    - rbf_kernel (bool): apply Gaussian (RBF) kernel to the matrix
    """
    softmax_client_weights = True
    p = copy.deepcopy(client.p)
    # print(len(p), len(p[0]))
    for i in range(len(p)):
        p[i] = np.array(p[i], dtype=np.float64)
        # print(p[i].type)
    matrix = np.zeros([len(p), len(p)], dtype=np.float64)
    for i in range(len(p)):
        # print(matrix[i].shape, p[i].shape)
        matrix[i] += p[i]
    # print(matrix.dtype)
    # print(type(matrix), type(matrix[0]))
    for i in range(len(matrix)):
        matrix[i] = np.array(matrix[i], dtype=np.float64)
    matrix = np.array(matrix)
    # print(matrix)
    if softmax_client_weights:
        # matrix = []
        # for client in clients:
        #     matrix.append(np.exp(client.w) /
        #                   np.sum(np.exp(client.w)))
        matrix = np.exp(matrix) / np.sum(np.exp(matrix))
        # matrix = np.array(matrix)
    else:
        # matrix = np.array([client.w for client in clients])
        # print(type(matrix), type(matrix[0]))
        matrix = 1. - matrix  # Affinity matrix is reversed -> lower value = better
        # matrix = np.exp(matrix)
        # print(matrix.dtype)
        if rbf_kernel:
            # print(-1. * matrix ** 2 / (2. * rbf_delta ** 2))
            # print(type(-1. * matrix ** 2 / (2. * rbf_delta ** 2)))
            matrix = np.exp(-1. * matrix ** 2 / (2. * rbf_delta ** 2))
    if symmetric:
        matrix = (matrix + matrix.T) * 0.5
    if shift_positive:
        if np.min(matrix) < 0:
            matrix = matrix + (0 - np.min(matrix))
    if normalize:
        # print("normalize")
        matrix = matrix / (np.max(matrix) - np.min(matrix))
    return matrix


# 计算emd热力图
def calc_emd_heatmap(train_data, train_idx_dict, args):
    client_num_in_total = args.client_num_in_total
    emd_list = np.zeros([client_num_in_total, client_num_in_total], dtype=np.float64)
    epsilon = 1e-10
    for c_id_from in range(client_num_in_total):
        for c_id_to in range(client_num_in_total):
            # if c_id_from == c_id_to:
            #     emd_list[c_id_from][c_id_to] = 0
            #     continue
            train_data_from = [train_data.targets[x] for x in train_idx_dict[c_id_from]]
            train_data_to = [train_data.targets[x] for x in train_idx_dict[c_id_to]]
            emd = compute_emd(train_data_from, train_data_to)
            # emd_list[c_id_from][c_id_to] = 1 / (emd + epsilon)
            emd_list[c_id_from][c_id_to] = emd
    # print(emd_list.shape)
    # emd_list = emd_list - emd_list.min() / (emd_list.max() - emd_list.min())
    # todo 这样写整个热力图色调偏冷，而权重热力图偏热，可能在下面那个循环改成倒数会好些
    emd_list = 1 - (emd_list - emd_list.min()) / (emd_list.max() - emd_list.min() + epsilon)
    for c_id in range(client_num_in_total):
        emd_list[c_id][c_id] = 0
    return emd_list


# 根据二维矩阵生成热力图
def generate_heatmap(matrix, path):
    # 显示数值 square=True, annot=True
    # print(len(matrix), len(matrix[0]))
    # print(matrix)
    plt.matshow(matrix, cmap=plt.cm.rainbow, vmin=0, vmax=1)
    plt.colorbar()
    # plt.show()
    # sns.heatmap(matrix, vmin=0, vmax=1, center=0.5)
    # # heatmap.get_figure().savefig(path, dpi=600)
    # print("===========用plt绘制")
    plt.savefig(path, dpi=600)
    plt.close()

# def getName(args):
#     name = ""
#     if args.model == "BaseConvNet":
#         name = "BCN"
#     else:
#         name = args.model
#     name += "-"+args.client_num_in_total+"c"
#     name += "-"+str(args.topology_neighbors_num_undirected)+"n"
#     name += "-"+str(args.epochs)+"e"
#     name += "-"+args.broadcaster_strategy
#     if args.local_train_stop_point <= args.comm_round:
#         name += "-"+str(args.local_train_stop_point)+"stop"
#     return name
def print_debug(stdout, prefix=''):
    DEBUG = True
    if DEBUG:
        print(f'DEBUG - {prefix}: {stdout}')

