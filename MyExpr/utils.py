import numpy as np
from collections import Counter

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

"""
计算本地模型与其他模型的权重
：local_para  本地模型参数
: parameters 其他模型参数集合
: return  权重列表
"""


# 计算本地模型与其他模型的权重
def CalW(lossFun, lr, train_ds, Net, local_para, eNet, parameters):

    w = []
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    Net.load_state_dict(local_para, strict=True)

    for para in parameters:
        # 两个模型loss
        loss1 = 0.0
        loss2 = 0.0

        # f为验证数据集的大小
        # 简单的确定合适的验证数据集大小
        f = 0
        with torch.no_grad():
            eNet.load_state_dict(para, strict=True)

            for data, label in train_dl:
                # 自己设置，大概10%-40%
                # 这是超参数
                if f == 10:
                    break
                f += 1
                # data, label = data.to(self.dev), label.to(self.dev)
                preds1 = Net(data)
                preds2 = eNet(data)

                loss1 += lossFun(preds1, label).item()
                loss2 += lossFun(preds2, label).item()

        # 计算分母
        dif = CalDif(local_para, para)
        if loss1 <= loss2:
            w.append(0)
        else:
            w.append((loss1 - loss2) / dif)
        # 未用到学习率lr,实际上,后面在对w进行归一化的时候，lr项是被约掉的
    return w


def CalDif(para1, para2):
    # 利用矩阵范数求模型间的差异
    # 比较粗糙的方式
    # 或者在算loss的时候，顺便把准确率算出来，作为模型间差异
    dif = 0
    for item in para1:
        dif += torch.norm(para1[item] - para2[item])
    return dif