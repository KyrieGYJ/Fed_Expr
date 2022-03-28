import numpy as np
from collections import Counter
import heapq
import seaborn as sns
import matplotlib.pyplot as plt
import torch
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

    for c_id, model in received_dic.items():
        # 计算分母
        dif = CalDif(local_model, model, args)
        w[c_id] = ((total_local_loss - loss_dic[c_id]) / dif)

        # # 计算分母
        # dif = CalDif(local_model, model)
        # # if loss1 <= loss2:
        # #     w[c_id] = 0
        # # else:
        # #     w[c_id] = ((loss1 - loss2) / dif)
        # # 类似于原文提到的，因为是用于计算广播价值，所以不需要去relu
        # w[c_id] = ((loss1 - loss2) / dif)
        # # 未用到学习率lr,实际上,后面在对w进行归一化的时候，lr项是被约掉的
    return w


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


# 根据二维矩阵生成热力图
def generate_heatmap(matrix, path):
    # 显示数值 square=True, annot=True
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

