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
from MyExpr.dfl.args import add_args
from MyExpr.dfl.component.client import Client
from MyExpr.dfl.component.top_k import TopKSelector
from MyExpr.dfl.component.broadcaster import Broadcaster
from MyExpr.dfl.component.trainer import Trainer
from MyExpr.dfl.component.recorder import Recorder
from MyExpr.data import Data

from MyExpr.utils import generate_heatmap
from MyExpr.utils import calc_emd_heatmap

# 固定种子后，deter设置为true能固定每次结果相同
# torch.backends.cudnn.deterministic = True
# cuda自动优化，重复同样网络会变快，反复修改网络会变慢
# torch.backends.cudnn.benchmark = False

parser = add_args()
args = parser.parse_args()

if "non-iid" in args.data_distribution:
    # 优先按比例划分
    args.num_clients_per_dist = int(args.client_num_in_total / args.num_distributions)

# 初始化组件
trainer = Trainer(args)
broadcaster = Broadcaster(args)
topK_selector = TopKSelector(args)
client_num_in_total = args.client_num_in_total
topology_manager = TopologyManager(client_num_in_total, True,
                                   undirected_neighbor_num=args.topology_neighbors_num_undirected)
topology_manager.generate_topology()
# 初始化数据
data = Data(args)
data.generate_loader()

client_dic = {}
recorder = Recorder(client_dic, topology_manager, trainer, broadcaster, topK_selector, data, args)

# 初始化client
model_builder = get_model_builder(args)
if args.enable_dp:
    print("开启差分隐私")
for c_id in tqdm(range(client_num_in_total), desc='setting up client'):
    c = Client(model_builder(num_classes=10), c_id, args, data, topK_selector, recorder, broadcaster)
    client_dic[c_id] = c
# 等client_dic完整后，在这里初始化affinity矩阵
# for c_id in range(client_num_in_total):
#     client_dic[c_id].initialize()

broadcaster.initialize()

name = None if args.name == '' else args.name

# 计算client数据之间的emd热力图
emd_list = calc_emd_heatmap(data.train_data, data.train_idx_dict, args)
if not os.path.exists(f'./heatmap/{name}'):
    os.makedirs(f'./heatmap/{name}')
generate_heatmap(emd_list, f"./heatmap/{name}/emd_heatmap2")

args.turn_on_wandb = True

# dfl6_20client_noniid
# dfl6_20client_iid
# dfl6
# dfl6_iid
if args.turn_on_wandb:
    wandb.init(project="weight_test",
               entity="kyriegyj",
               name=name,
               config=args)

#################################
# local train before federation #
#################################
precomputed = False

if precomputed:
    print("use pretrained model")
    # model_dict_fname = f"./precomputed/{args.model}_{args.client_num_in_total}"
    model_dict_fname = f"./precomputed/{args.model}_{args.client_num_in_total}_pathological2"
    model_dict = {i : None for i in range(args.client_num_in_total)}

    try:
        model_dict = torch.load(model_dict_fname)
        print(f'model dictionary model loaded from {model_dict_fname}')
    except:
        print(f'no model dictionary was found, newly train {model_dict_fname}')
        pretrain_epoch = 30
        for E in range(pretrain_epoch):
            trainer.local(False)
        for i in range(args.client_num_in_total):
            model_dict[i] = client_dic[i].model
        torch.save(model_dict, model_dict_fname)

    for i in range(args.client_num_in_total):
        client_dic[i].model= model_dict[i]

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
    # trainer.overall_test()
    # 所属的distribution测试数据上测试
    # if "non-iid" in args.data_distribution:
    #     trainer.non_iid_test()
    # 打印affinity热力图（如果采用聚类广播算法）
    broadcaster.get_w_heatmap(f"./heatmap/{name}/weight_{rounds}")
    # 打印p矩阵热力图
    # if args.broadcaster_strategy == "affinity_topK":
    #     broadcaster.get_clients_p_heatmap(f"./heatmap/{name}/p_{rounds}")
    # 打印通信频率热力图
    broadcaster.get_freq_heatmap(f"./heatmap/{name}/freq_{rounds}")
    print("-----第{}轮训练结束-----".format(rounds))


# 保存模型
if args.turn_on_wandb:
    client_model_dic = {}
    for id in client_dic:
        client = client_dic[id]
        client_model_dic[id] = client.model
    torch.save(client_model_dic, "./model/{:s}_client_dic".format(name))

if args.turn_on_wandb:
    wandb.finish()