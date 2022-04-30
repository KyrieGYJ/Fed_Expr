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


def eval(model, client, args):
    criterion = client.criterion_CE
    val_dataloader = client.validation_loader
    epsilon = 1e-6
    total_local_loss = 0.0
    total_local_correct = 0.0

    with torch.no_grad():
        model.eval()
        model.to(args.device)
        for data, label in val_dataloader:
            data, label = data.to(args.device), label.to(args.device)
            local_outputs = model(data)
            local_loss = criterion(local_outputs, label)

            pred = local_outputs.argmax(dim=1)
            correct = pred.eq(label.view_as(pred)).sum()

            if "cuda" in args.device:
                local_loss = local_loss.cpu()
                correct = correct.cpu()

            total_local_loss += local_loss.item()
            total_local_correct += correct
    return total_local_loss, total_local_correct


def calc_eval(client, received_model_dict, args):
    """
    评估客户本身以及接收到的模型
    ：client  客户
    : received_model_dict 接收到的模型，自己设置，可以比较灵活
    : return  eval_loss_dict, eval_acc_dict[包含本地模型]
    """
    # 因为在local_update后broadcast，采用本轮local_update后的模型和上一轮接收到的模型进行计算
    # 相当于基于接受者上一轮的模型预测：接受者的本地模型在本地经历过mutual_update和新一轮local_update后，发送者的新本地模型和它的仍然是相近的。
    criterion = client.criterion_CE
    local_model = client.model

    total_local_loss, total_local_correct = 0.0, 0.0
    loss_dict, acc_dict = {}, {}
    for c_id, model in received_model_dict.items():
        loss_dict[c_id], acc_dict[c_id] = 0.0, 0.0

    with torch.no_grad():
        local_model.eval()
        local_model.to(args.device)
        for data, label in client.validation_loader:
            data, label = data.to(args.device), label.to(args.device)
            local_outputs = local_model(data)
            local_loss = criterion(local_outputs, label)
            pred = local_outputs.argmax(dim=1)
            correct = pred.eq(label.view_as(pred)).sum()
            total_local_loss += local_loss.cpu().item()
            total_local_correct += correct.cpu()

            for c_id, model in received_model_dict.items():
                if c_id == client.client_id:
                    continue
                model.eval()
                model.to(args.device)
                received_outputs = model(data)
                received_loss = criterion(received_outputs, label)
                pred = received_outputs.argmax(dim=1)
                correct = pred.eq(label.view_as(pred)).sum()
                loss_dict[c_id] += received_loss.cpu().item()
                acc_dict[c_id] += correct.cpu()
    loss_dict[client.client_id] = total_local_loss
    acc_dict[client.client_id] = total_local_correct
    for c_id, correct in acc_dict.items():
        acc_dict[c_id] = correct / len(client.validation_set)
    return loss_dict, acc_dict


def calc_eval_speed_up_using_cache(cache_keeper):
    args = cache_keeper.args
    host = cache_keeper.host
    criterion = host.criterion_CE
    loss_dict, acc_dict = {}, {}
    # new_model_dict = {host.client_id:host.model}
    new_model_dict = host.received_model_dict

    known_set = cache_keeper.known_set

    for c_id, model in new_model_dict.items():
        loss_dict[c_id], acc_dict[c_id] = 0.0, 0.0

    # 计算新接收到的模型(包括当前的host model)
    # print(f"received model:{host.received_model_dict.keys()}")
    with torch.no_grad():
        for data, label in host.validation_loader:
            data, label = data.to(args.device), label.to(args.device)
            for c_id, model in new_model_dict.items():
                model.eval()
                model.to(args.device)
                received_outputs = model(data)
                received_loss = criterion(received_outputs, label)
                pred = received_outputs.argmax(dim=1)
                correct = pred.eq(label.view_as(pred)).sum()
                loss_dict[c_id] += received_loss.cpu().item()
                acc_dict[c_id] += correct.cpu()


    # 用memory填充未接收到的部分
    for i in range(args.client_num_in_total):
        if i not in new_model_dict and i in known_set:
            loss_dict[i] = cache_keeper.raw_eval_loss_list[i]
            acc_dict[i] = cache_keeper.raw_eval_acc_list[i]

    # correct转acc
    for c_id, correct in acc_dict.items():
        acc_dict[c_id] = correct / len(host.validation_set)
    return loss_dict, acc_dict


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


def calc_emd_heatmap(train_data, train_idx_dict, args):
    """
    Args:
        train_data: 总训练数据集
        train_idx_dict: client划分训练数据的字典
        args:
    Returns:
        反应数据集之间相似度的热力图矩阵
    """
    client_num_in_total = args.client_num_in_total - args.malignant_num
    emd_list = np.zeros([client_num_in_total, client_num_in_total], dtype=np.float64)
    epsilon = 1e-10
    for c_id_from in range(client_num_in_total):
        for c_id_to in range(client_num_in_total):
            # pytorch1.10
            # train_data_from = [train_data.targets[x] for x in train_idx_dict[c_id_from]]
            # train_data_to = [train_data.targets[x] for x in train_idx_dict[c_id_to]]
            # pytorch 1.8
            train_data_from = [train_data.train_labels[x] for x in train_idx_dict[c_id_from]]
            train_data_to = [train_data.train_labels[x] for x in train_idx_dict[c_id_to]]
            emd = compute_emd(train_data_from, train_data_to)
            emd_list[c_id_from][c_id_to] = emd
    emd_list = 1 - (emd_list - emd_list.min()) / (emd_list.max() - emd_list.min() + epsilon)
    return emd_list


# 根据二维矩阵生成热力图
def generate_heatmap(matrix, path, vmin=0, vmax=1):
    # 显示数值 square=True, annot=True
    plt.matshow(matrix, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
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