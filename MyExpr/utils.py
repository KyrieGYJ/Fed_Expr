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


def calc_eval(client, received_model_dict, args, loss_only = True):
    """
    计算本地模型与其他模型的权重
    ：local_para  本地模型参数
    : parameters 其他模型参数集合
    : return  权重列表
    """
    # 因为在local_update后broadcast，采用本轮local_update后的模型和上一轮接收到的模型进行计算
    # 相当于基于接受者上一轮的模型预测：接受者的本地模型在本地经历过mutual_update和新一轮local_update后，发送者的新本地模型和它的仍然是相近的。
    w_dict = {}
    criterion = client.criterion_CE
    val_dataloader = client.validation_loader
    local_model = client.model
    # received_dic = client.last_received_model_dict
    # received_dict = client.received_model_dict

    epsilon = 1e-6
    total_local_loss = 0.0
    total_local_correct = 0.0
    loss_dict = {}
    acc_dict = {}
    for c_id, model in received_model_dict.items():
        loss_dict[c_id] = 0.0
        acc_dict[c_id] = 0

    with torch.no_grad():
        local_model.eval()
        for data, label in val_dataloader:
            data, label = data.to(args.device), label.to(args.device)
            local_outputs = local_model(data)
            local_loss = criterion(local_outputs, label)

            pred = local_outputs.argmax(dim=1)
            correct = pred.eq(label.view_as(pred)).sum()

            if "cuda" in args.device:
                local_loss = local_loss.cpu()
                correct = correct.cpu()

            total_local_loss += local_loss.item()
            total_local_correct += correct

            for c_id, model in received_model_dict.items():
                model.eval()
                received_outputs = model(data)
                received_loss = criterion(received_outputs, label)

                pred = received_outputs.argmax(dim=1)
                correct = pred.eq(label.view_as(pred)).sum()

                if "cuda" in args.device:
                    received_loss = received_loss.cpu()
                    correct = correct.cpu()

                loss_dict[c_id] += received_loss.item()
                acc_dict[c_id] += correct

    loss_dict[client.client_id] = total_local_loss
    acc_dict[client.client_id] = total_local_correct
    for c_id, correct in acc_dict.items():
        acc_dict[c_id] = correct / len(client.validation_set)
    return loss_dict, acc_dict


# 计算模型之间的权重(参考ICLR FedFomo)
def calc_delta_loss(client, received_model_dict, args, loss_only = True):
    """
    计算本地模型与其他模型的权重
    ：local_para  本地模型参数
    : parameters 其他模型参数集合
    : return  权重列表
    """
    # 因为在local_update后broadcast，采用本轮local_update后的模型和上一轮接收到的模型进行计算
    # 相当于基于接受者上一轮的模型预测：接受者的本地模型在本地经历过mutual_update和新一轮local_update后，发送者的新本地模型和它的仍然是相近的。
    w_dict = {}
    criterion = client.criterion_CE
    val_dataloader = client.validation_loader
    local_model = client.model
    # received_dic = client.last_received_model_dict
    # received_dict = client.received_model_dict

    epsilon = 1e-6
    total_local_loss = 0.0
    loss_dict = {}
    for c_id, model in received_model_dict.items():
        loss_dict[c_id] = 0.0

    with torch.no_grad():
        local_model.eval()
        for data, label in val_dataloader:
            data, label = data.to(args.device), label.to(args.device)
            local_outputs = local_model(data)
            local_loss = criterion(local_outputs, label)

            if "cuda" in args.device:
                local_loss = local_loss.cpu()

            total_local_loss += local_loss.item()

            for c_id, model in received_model_dict.items():
                model.eval()
                received_outputs = model(data)
                received_loss = criterion(received_outputs, label)

                if "cuda" in args.device:
                    received_loss = received_loss.cpu()

                loss_dict[c_id] += received_loss.item()

    for c_id, model in received_model_dict.items():
        # 计算分母
        w_dict[c_id] = total_local_loss - loss_dict[c_id]

    return w_dict


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


# def get_adjacency_matrix_decrapted(client, symmetric=True, shift_positive=False,
#                          normalize=True, rbf_kernel=False, rbf_delta=1.):
#     """
#     If learning federations through updated client weight preferences,
#     first compute adjacency matrix given client-to-client weights
#     Args:
#     - clients (Clients[]): list of clients <-- should be the population.clients array
#     - symmetric (bool): return a symmetric matrix
#     - shift_positive (bool): shift all values to be >= 0 (add (0 - lowest value) to everything)
#     - normalize (bool): after other transformations, normalize everything to range [0, 1]
#     - rbf_kernel (bool): apply Gaussian (RBF) kernel to the matrix
#     """
#     # 不能用softmax， 太低了
#     softmax_client_weights = False
#     p = copy.deepcopy(client.p)
#     matrix = np.array(p)
#     # print(matrix)
#     if softmax_client_weights:
#         matrix = np.exp(matrix) / np.sum(np.exp(matrix))
#     else:
#         matrix = 1. - matrix  # Affinity matrix is reversed -> lower value = better # 谜之公式
#         if rbf_kernel:
#             # print(-1. * matrix ** 2 / (2. * rbf_delta ** 2))
#             # print(type(-1. * matrix ** 2 / (2. * rbf_delta ** 2)))
#             matrix = np.exp(-1. * matrix ** 2 / (2. * rbf_delta ** 2)) # 谜之公式
#     if symmetric:
#         matrix = (matrix + matrix.T) * 0.5
#     if shift_positive:
#         if np.min(matrix) < 0:
#             matrix = matrix + (0 - np.min(matrix))
#     if normalize:
#         # print("normalize")
#         # 也不能用normalize，会出现infs or NaNs
#         matrix = matrix / (np.max(matrix) - np.min(matrix))
#     return matrix


def get_adjacency_matrix(client, softmax_client_weights=False, symmetric=True, shift_positive=False,
                         normalize=True, rbf_kernel=False, rbf_delta=1.):
    p = copy.deepcopy(client.p)
    matrix = np.array(p)
    # 不能用因为矩阵总的exp和比较大，相除后每一项会特别小
    if softmax_client_weights:
        matrix = np.exp(matrix) / np.sum(np.exp(matrix))
        # print(f"softmax matrix{matrix}")
    if symmetric:
        matrix = (matrix + matrix.T) * 0.5
    # 开启后矩阵中的值会非常高，整个图偏红
    if shift_positive:
        if np.min(matrix) < 0:
            matrix = matrix + (0 - np.min(matrix))
        # print(f"shift positive matrix{matrix}"

    # 调整自身权重为最大
    for i in range(len(matrix)):
        matrix[i][i] = np.min(matrix)
    for i in range(len(matrix)):
        matrix[i][i] = np.max(matrix)

    if normalize:
        # matrix = matrix - np.min(matrix) / (np.max(matrix) - np.min(matrix))
        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

    return matrix


def calc_emd_heatmap(train_data, train_idx_dict, args):
    """
    Args:
        train_data: 总训练数据集
        train_idx_dict: client划分训练数据的字典
        args:
    Returns:
        反应数据集之间相似度的热力图矩阵
    """
    client_num_in_total = args.client_num_in_total
    emd_list = np.zeros([client_num_in_total, client_num_in_total], dtype=np.float64)
    epsilon = 1e-10
    for c_id_from in range(client_num_in_total):
        for c_id_to in range(client_num_in_total):
            train_data_from = [train_data.targets[x] for x in train_idx_dict[c_id_from]]
            train_data_to = [train_data.targets[x] for x in train_idx_dict[c_id_to]]
            emd = compute_emd(train_data_from, train_data_to)
            emd_list[c_id_from][c_id_to] = emd
    # todo 这样写整个热力图色调偏冷，而权重热力图偏热，可能在下面那个循环改成倒数会好些
    emd_list = 1 - (emd_list - emd_list.min()) / (emd_list.max() - emd_list.min() + epsilon)
    return emd_list


# 根据二维矩阵生成热力图
def generate_heatmap(matrix, path):
    # 显示数值 square=True, annot=True
    plt.matshow(matrix, cmap=plt.cm.rainbow, vmin=np.min(matrix), vmax=np.max(matrix))
    plt.colorbar()
    # plt.show()
    # sns.heatmap(matrix, vmin=0, vmax=1, center=0.5)
    # # heatmap.get_figure().savefig(path, dpi=600)
    # print("===========用plt绘制")
    plt.savefig(path, dpi=600)
    plt.close()


def print_debug(stdout, prefix=''):
    DEBUG = True
    if DEBUG:
        print(f'DEBUG - {prefix}: {stdout}')

