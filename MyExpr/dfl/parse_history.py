import logging
import torch
from tqdm import tqdm
import wandb
import os
import sys
import time
import matplotlib.pyplot as plt

# 添加环境
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../MyExpr")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

project_name = "test2"
name = "c10_pfedme_at_latent2_nd5_bk0.1"
history = torch.load(f"./BKW/{project_name}/{name}/history", map_location='cpu')

comm_round = history.args.comm_round
dist_num = history.args.num_distributions
client_num = history.args.client_num_in_total


test = False
only_first = True

color = ["red", "darkorange", "greenyellow", "aquamarine", "teal"]
x = [i for i in range(comm_round)]

if not test:
    print("parse broadcast weight")
    weight_dict = history.broadcast_weight_history_dict
    if weight_dict is not None:
        for i in weight_dict:
            bkw = weight_dict[i]
            plt.title(f'client {i}')
            plt.xlabel('comm_round')  # x轴标题
            plt.ylabel('bw')  # y轴标题
            legend = []
            for i in range(client_num):
                y_i = bkw[..., i]
                plt.plot(x, y_i, color=color[int(i / (client_num / dist_num))])
                legend.append(f"client {i}")
            plt.legend(legend, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, prop={'size': 6})
            # plt.gcf().subplots_adjust(left=0.05, right=0.8, top=0.91, bottom=0.09)
            plt.savefig(f"./BKW/{project_name}/{name}/pic", dpi=600, bbox_inches='tight')  # 自动计算bbox使得包含整个图
            plt.close()
            if only_first:
                break
    print("parse loss && acc")
    # loss && acc
    global_title = ""
    global_history_type = ["train_loss", "train_acc", "test_loss", "test_acc"]
    for t in global_history_type:
        history_box = history.get_history_box(t)
        if history_box is None:
            continue
        plt.title('global_title')
        plt.xlabel('comm_round')  # x轴标题
        plt.ylabel(t)  # y轴标题
        plt.plot(x, history_box)
        plt.savefig(f"./BKW/{project_name}/{name}/{t}", dpi=600, bbox_inches='tight')  # 自动计算bbox使得包含整个图
        plt.close()
    # heatmap
    print("parse heatmap")
    if not os.path.exists(f'./BKW/{project_name}/{name}/heatmap'):
        os.makedirs(f'./BKW/{project_name}/{name}/heatmap')
    freq_maps = history.broadcast_freq
    if freq_maps is not None:
        for i in range(comm_round):
            freq_map = freq_maps[i]
            freq = (freq_map - freq_map.min()) / (freq_map.max() - freq_map.min() + 1e-6)
            # 显示数值 square=True, annot=True
            plt.matshow(freq_map, cmap=plt.cm.rainbow, vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(f"./BKW/{project_name}/{name}/heatmap/freq_{i}", dpi=600)
            plt.close()

    # plt.show()
    # sns.heatmap(matrix, vmin=0, vmax=1, center=0.5)
    # # heatmap.get_figure().savefig(path, dpi=600)
    # print("===========用plt绘制")
else:
    print(history.best_accuracy)


# bkw = torch.load(f"./BKW/{project_name}/{name}/weight_arr")
# plt.title('client 0')  # 折线图标题
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
# plt.xlabel('comm_round')  # x轴标题
# plt.ylabel('bw')  # y轴标题
# legend = []
# for i in range(client_num):
#     y_i = bkw[..., i]
#     plt.plot(x, y_i, color=color[int(i / (client_num / dist_num))])
#     legend.append(f"client {i}")
# plt.legend(legend, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, prop={'size': 6})
# # plt.gcf().subplots_adjust(left=0.05, right=0.8, top=0.91, bottom=0.09)
# plt.savefig(f"./BKW/{project_name}/{name}/pic", dpi=600, bbox_inches='tight')  # 自动计算bbox使得包含整个图
# plt.close()