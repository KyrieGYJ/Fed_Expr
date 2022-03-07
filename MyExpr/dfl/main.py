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
trainer = Trainer(args.mutual_trainer_strategy)
trainer.use("local_and_mutual")
# 2、设置broadcaster策略
broadcaster = Broadcaster(args.broadcaster_strategy)
broadcaster.use("flood")
# 3、设置Top_K策略
topK_selector = TopKSelector(args.topK_strategy)
topK_selector.use("loss")
# 4、初始化拓扑结构
client_num_in_total = args.client_num_in_total
topology_manager = TopologyManager(client_num_in_total, True,
                                           undirected_neighbor_num=args.topology_neighbors_num_undirected)
# 5、加载数据集，划分
data = Data(args)
train_loader, test_loader, test_all = data.train_loader, data.test_loader, data.test_all
train_data_size_per_client = int(len(train_loader) / client_num_in_total)
test_data_size_per_client = int(len(test_loader) / client_num_in_total)
epochs = args.epochs
batch_size = args.batch_size
train_iteration = int(train_data_size_per_client / batch_size)


client_dic = {}
# 6、注册recorder
recorder = Recorder(client_dic, topology_manager)


# 7、初始化client, 选择搭载模型等
for c_id in range(client_num_in_total):
    # "ResNet18_GN"
    model = resnet18(num_classes=10)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.wd, amsgrad=True)
    c = Client(model, c_id, trainer, args, train_loader[c_id], test_loader[c_id])
    # 方便更换策略
    c.register(topK_selector=topK_selector, recorder=recorder, broadcaster=broadcaster)
    client_dic[c_id] = c


# 8、开始训练
for epoch in range(epochs):
    # train
    for iteration in range(train_iteration):
        logging.info("============开始训练(第:d轮)============".format(iteration))
        trainer.train(iteration)
        logging.info("============结束训练(第:d轮)============".format(iteration))
    # todo 输出train_loss
    # todo test



