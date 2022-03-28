import logging
import torch
from tqdm import tqdm
import wandb
import os
import sys

import numpy as np


# 添加环境
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../MyExpr")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from fedml_api.standalone.decentralized.topology_manager import TopologyManager

from MyExpr.dfl.model.model_builder import get_model_builder
from MyExpr.dfl.Args import add_args
from MyExpr.dfl.component.client import Client
from MyExpr.dfl.component.top_k import TopKSelector
from MyExpr.dfl.component.broadcaster import Broadcaster
from MyExpr.dfl.component.trainer import Trainer
from MyExpr.dfl.component.recorder import Recorder
from MyExpr.data import Data

from MyExpr.utils import compute_emd
from MyExpr.utils import generate_heatmap

# 固定种子后，deter设置为true能固定每次结果相同
# torch.backends.cudnn.deterministic = True
# cuda自动优化，重复同样网络会变快，反复修改网络会变慢
# torch.backends.cudnn.benchmark = False

parser = add_args()
args = parser.parse_args()
trainer = Trainer(args)
print("初始化trainer完成")
broadcaster = Broadcaster(args)
print("初始化broadcaster完成")
topK_selector = TopKSelector(args)
print("初始化topK_selector完成")
client_num_in_total = args.client_num_in_total
print(f"总共{client_num_in_total}个client进行试验")
print("初始化component...完成")
# 初始化拓扑结构
print("**********generating topology**********")
topology_manager = TopologyManager(client_num_in_total, True,
                                   undirected_neighbor_num=args.topology_neighbors_num_undirected)
topology_manager.generate_topology()
print("**********finishing topology generation**********")

# 5、加载数据集，划分
data = Data(args)
data.generate_loader()

trainer.test_loader = data.test_all

# 临时这么写方便，todo 后续有需要再改
if args.data_distribution == "non-iid_pathological":
    test_non_iid, client_class_dic, class_client_dic= data.test_non_iid, data.client_class_dic, data.class_client_dic
    # print("length of test_all: {}".format(len(test_all)))
    # print("length of test_non_iid: {}".format(len(test_non_iid[0])))
    # todo 这一堆初始化都要改掉
    trainer.test_non_iid = data.test_non_iid
    trainer.client_class_dic = data.client_class_dic
    trainer.class_client_dic = data.class_client_dic
elif args.data_distribution == "non-iid_latent":
    # 加载出non-iid数据字典
    trainer.dist_client_dict = data.dist_client_dict
    trainer.test_non_iid = data.test_non_iid

client_dic = {}
print("初始化recorder...", end="")
recorder = Recorder(client_dic, topology_manager, trainer, broadcaster, topK_selector, args)
print("完毕")

# 选择网络
model_builder = get_model_builder(args)
print(f"采用网络{args.model}进行试验")

# 7、初始化client, 选择搭载模型等
print("初始化clients...", end="")
train_loader, validation_loader, test_loader = data.train_loader, data.validation_loader, data.test_loader
for c_id in range(client_num_in_total):
    # "ResNet18_GN"
    model = model_builder(num_classes=10)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.wd, amsgrad=True)
    c = Client(model, c_id, args, train_loader[c_id], validation_loader[c_id], test_loader[c_id])
    # 方便更换策略
    c.register(topK_selector=topK_selector, recorder=recorder, broadcaster=broadcaster)
    client_dic[c_id] = c
print("完毕")

broadcaster.initialize()

name = None if args.name == '' else args.name

# 计算client数据之间的emd热力图
emd_list = np.zeros([client_num_in_total, client_num_in_total], dtype = np.float64)
train_data = data.train_data
train_idx_dict = data.train_idx_dict
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

emd_list = 1 - (emd_list - emd_list.min()) / (emd_list.max() - emd_list.min() + epsilon)
for c_id in range(client_num_in_total):
    emd_list[c_id][c_id] = 0
# print(emd_list)
if not os.path.exists(f'./heatmap/{name}'):
    os.makedirs(f'./heatmap/{name}')
generate_heatmap(emd_list, f"./heatmap/{name}/emd_heatmap2")

args.turn_on_wandb = False

if args.turn_on_wandb:
    wandb.init(project="dfl4",
               entity="kyriegyj",
               name=name,
               config=args)

###############################
# 1 communication per E epoch #
###############################
for rounds in range(args.comm_round):
    print("-----开始第{}轮训练-----".format(rounds))
    recorder.rounds = rounds
    trainer.train()
    # 在本地数据集上测试
    trainer.local_test()
    # 在全局数据集上测试
    trainer.overall_test()
    # 在每个类包含的标签的测试数据上测试
    trainer.non_iid_test()
    # 打印权重热力图（如果采用聚类广播算法）
    broadcaster.get_p_heatmap(f"./heatmap/{name}/weight_{rounds}")
    # 打印通信频率热力图
    broadcaster.get_freq_heatmap(f"./heatmap/{name}/freq_{rounds}")

    print("-----第{}轮训练结束-----".format(rounds))


# 9、保存模型
if args.turn_on_wandb:
    client_model_dic = {}
    for id in client_dic:
        client = client_dic[id]
        client_model_dic[id] = client.model
    torch.save(client_model_dic, "./model/{:s}_client_dic".format(name))

if args.turn_on_wandb:
    wandb.finish()
