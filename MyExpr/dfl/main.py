import logging
import torch
from tqdm import tqdm
import wandb
import os
import sys
import time

# 添加环境
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../MyExpr")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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


def initialize(args):
    malignant_num = 20
    print(f"使用设备:{args.device}")
    print(f"恶意节点数量:{malignant_num}")
    # 初始化数据
    data = Data(args)
    data.generate_loader()
    # 初始化组件
    trainer = Trainer(args)
    broadcaster = Broadcaster(args)
    topK_selector = TopKSelector(args)

    client_num_in_total = args.client_num_in_total
    args.malignant_num = malignant_num
    args.client_num_in_total = client_num_in_total + malignant_num

    # todo 下一版删掉，本实验不考虑
    topology_manager = TopologyManager(client_num_in_total + malignant_num, True,
                                       undirected_neighbor_num=args.topology_neighbors_num_undirected)
    topology_manager.generate_topology()

    client_dict = {}
    recorder = Recorder(client_dict, topology_manager, trainer, broadcaster, topK_selector, data, args)

    # 初始化client
    model_builder = get_model_builder(args)
    if args.enable_dp:
        print("开启差分隐私")
    for c_id in tqdm(range(client_num_in_total), desc='setting up client'):
        c = Client(model_builder(num_classes=10), c_id, args, data, topK_selector, recorder, broadcaster)
        client_dict[c_id] = c

    # 添加恶意节点 todo 恶意节点可能利用意外接受到的模型。
    malignant_dict = {}
    for c_id in tqdm(range(malignant_num)):
        c = Client(model_builder(num_classes=10), c_id + client_num_in_total, args, data, topK_selector, recorder, broadcaster)
        malignant_dict[c_id + client_num_in_total] = c
    trainer.malignant_dict = malignant_dict
    recorder.malignant_dict = malignant_dict

    broadcaster.initialize()
    return client_dict, data, trainer, recorder, broadcaster


def main():
    parser = add_args()
    args = parser.parse_args()
    if "non-iid" in args.data_distribution:
        # 优先按比例划分
        args.num_clients_per_dist = int(args.client_num_in_total / args.num_distributions)

    client_dict, data, trainer, recorder, broadcaster = initialize(args)

    name = None if args.name == '' else args.name
    project_name = None if args.project_name == '' else args.project_name
    # 计算client数据之间的emd热力图
    emd_list = calc_emd_heatmap(data.train_data, data.train_idx_dict, args)
    if not os.path.exists(f'./heatmap/{project_name}/{name}'):
        os.makedirs(f'./heatmap/{project_name}/{name}')
    generate_heatmap(emd_list, f"./heatmap/{project_name}/{name}/emd_heatmap")

    args.turn_on_wandb = name is not None and project_name is not None and False

    if args.pretrain_epoch > 0:
        model_dict_fname = f"./precomputed/pretrain/{args.model}_c{args.client_num_in_total - args.malignant_num}" \
                           f"_{args.data_distribution}_dn{args.num_distributions}_pe{args.pretrain_epoch}"
        model_dict = {i: None for i in range(args.client_num_in_total)}
        try:
            # 这里不加CPU会把它加载到原先训练的GPU上。
            model_dict = torch.load(model_dict_fname, map_location='cpu')
            print(f'> model dictionary {model_dict_fname} exists, no need to compute')
        except Exception as e:
            print(e)
            print(f"> no model dictionary {model_dict_fname} exists, please pretrain")
            return
        for i in client_dict:
            client_dict[i].model.load_state_dict(model_dict[i])


    if args.turn_on_wandb:
        wandb.init(project=project_name,
                   entity="kyriegyj",
                   name=name,
                   config=args)



    ###############################
    # 1 communication per E epoch #
    ###############################
    for rounds in range(args.comm_round):
        start = time.time()
        print(f"-----开始第{rounds}轮训练-----")
        recorder.rounds = rounds
        trainer.train()
        # 在本地数据集上测试
        trainer.local_test()
        # 打印通信频率热力图
        broadcaster.get_freq_heatmap(f"./heatmap/{project_name}/{name}/freq_{rounds}")
        end = time.time()
        print(f"-----第{rounds}轮训练结束, 耗时:{end - start}s-----")

    # save_model
    fname = f"{args.model}_c{args.client_num_in_total}" \
            f"_{args.data_distribution}_dn{args.num_distributions}"
    if args.pretrain_epoch > 0:
        fname = f"{fname}_pe{args.pretrain_epoch}"
    model_dict = {}
    for c_id in client_dict:
        model_dict[c_id] = client_dict[c_id].model
    if not os.path.exists(f'./model/{project_name}'):
        os.makedirs(f'./model/{project_name}')
    torch.save(model_dict, f"./model/{project_name}/{fname}")

    if args.turn_on_wandb:
        wandb.finish()


if __name__ == '__main__':
    test = False
    if test:
        print("测试脚本")
    else:
        print(f"当前pid: {os.getpid()}")
        main()
