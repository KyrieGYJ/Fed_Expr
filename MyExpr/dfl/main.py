import logging
import torch
from tqdm import tqdm

from fedml_api.standalone.decentralized.topology_manager import TopologyManager

from MyExpr.dfl.model.resnet import resnet18
from MyExpr.dfl.Args import add_args
from MyExpr.dfl.component.client import Client
from MyExpr.dfl.component.top_k import TopKSelector
from MyExpr.dfl.component.broadcaster import Broadcaster
from MyExpr.dfl.component.trainer import Trainer
from MyExpr.dfl.component.recorder import Recorder
from MyExpr.data import Data

parser = add_args()
args = parser.parse_args()

# 1、设置trainer策略
trainer = Trainer()
trainer.use(args.mutual_trainer_strategy)
# 2、设置broadcaster策略
broadcaster = Broadcaster()
broadcaster.use(args.broadcaster_strategy)
# 3、设置Top_K策略
topK_selector = TopKSelector()
topK_selector.use(args.topK_strategy)
# 4、初始化拓扑结构
client_num_in_total = args.client_num_in_total
topology_manager = TopologyManager(client_num_in_total, True,
                                           undirected_neighbor_num=args.topology_neighbors_num_undirected)
# 5、加载数据集，划分
data = Data(args)
train_loader, test_loader, test_all = data.train_loader, data.test_loader, data.test_all
train_data_size_per_client = len(train_loader[0])
test_data_size_per_client = len(test_loader[0])
epochs = args.epochs
batch_size = args.batch_size
train_iteration = int(train_data_size_per_client / batch_size)


client_dic = {}
# 6、注册recorder
recorder = Recorder(client_dic, topology_manager, args)
trainer.register_recorder(recorder)
broadcaster.register_recorder(recorder)
topK_selector.register_recoder(recorder)

# 7、初始化client, 选择搭载模型等
for c_id in range(client_num_in_total):
    # "ResNet18_GN"
    model = resnet18(num_classes=10)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.wd, amsgrad=True)
    c = Client(model, c_id, args, train_loader[c_id], test_loader[c_id])
    # 方便更换策略
    c.register(topK_selector=topK_selector, recorder=recorder, broadcaster=broadcaster)
    client_dic[c_id] = c

# todo 插入 wandb

# 8、开始训练
for epoch in range(epochs):
    # train
    for iteration in range(train_iteration):
        logging.info("============开始训练(第:d轮)============".format(iteration))
        trainer.train()
        logging.info("============结束训练(第:d轮)============".format(iteration))
    trainer.next_epoch()
    recorder.next_epoch()
    break
    # todo test



