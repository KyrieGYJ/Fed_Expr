import os
import numpy as np
import copy
import plt


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


def get_clients_p_heatmap(self, path):
    print("绘制p矩阵热力图")
    for c_id in range(self.args.client_num_in_total):
        p_list = self.recorder.client_dic[c_id].p
        if p_list is None or p_list == []:
            continue
        if not os.path.exists(path):
            os.makedirs(path)
        generate_heatmap(p_list, f"{path}/client_{c_id}")


def pass_heatmap(self, path):
    pass


def get_freq_heatmap(self, path):
    if self.strategy == "flood":
        return
    epsilon = 1e-6
    n = self.args.client_num_in_total
    freq = copy.deepcopy(self.broadcast_freq)
    for c_id in range(n):
        freq[c_id][c_id] = freq.max()
    freq = (freq - freq.min()) / (freq.max() - freq.min() + epsilon)
    print("绘制通信频率热力图")
    generate_heatmap(freq, path)