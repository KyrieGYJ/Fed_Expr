import logging
import torch
from tqdm import tqdm
import wandb
import os
import sys

# 添加环境
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../MyExpr")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

print(sys.path)

from fedml_api.standalone.decentralized.topology_manager import TopologyManager

from MyExpr.dfl.model.resnet import resnet18
from MyExpr.dfl.model.resnet import resnet34
from MyExpr.dfl.model.resnet import resnet50
from MyExpr.dfl.model.resnet import resnet101
from MyExpr.dfl.model.resnet import resnet152
from MyExpr.dfl.model.cnn import BaseConvNet
from MyExpr.dfl.Args import add_args
from MyExpr.dfl.component.client import Client
from MyExpr.dfl.component.top_k import TopKSelector
from MyExpr.dfl.component.broadcaster import Broadcaster
from MyExpr.dfl.component.trainer import Trainer
from MyExpr.dfl.component.recorder import Recorder
from MyExpr.data import Data

parser = add_args()
args = parser.parse_args()

# 查看GPU
print(torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print("GPU[{:d}]: {:s}".format(i, torch.cuda.get_device_name(i)))

print("当前模式：{}".format(args.communication_wise))
# 选择GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 设置component，只表示策略，都是单例
# 1、设置trainer策略
trainer = Trainer(args)
trainer.use(args.trainer_strategy)
# 2、设置broadcaster策略
broadcaster = Broadcaster()
broadcaster.use(args.broadcaster_strategy)
# 3、设置Top_K策略
topK_selector = TopKSelector(args)
topK_selector.use(args.topK_strategy)
# 4、初始化拓扑结构
client_num_in_total = args.client_num_in_total
topology_manager = TopologyManager(client_num_in_total, True,
                                   undirected_neighbor_num=args.topology_neighbors_num_undirected)
topology_manager.generate_topology()
print("finished topology generation")

print("Data:", os.path.abspath(os.path.join(os.getcwd(), args.data_dir)))

# 5、加载数据集，划分
data = Data(args)
data.generate_loader()
train_loader, validation_loader, test_loader, test_all = data.train_loader, data.validation_loader, data.test_loader, data.test_all
train_data_size_per_client = len(train_loader[0])
test_data_size_per_client = len(test_loader[0])

epochs = args.epochs
batch_size = args.batch_size
train_iteration = train_data_size_per_client
trainer.train_iteration = train_iteration
trainer.batch_size = batch_size

trainer.set_test_loader(test_all)

client_dic = {}
# 6、注册recorder
print("注册recorder")
recorder = Recorder(client_dic, topology_manager, args)
trainer.register_recorder(recorder)
broadcaster.register_recorder(recorder)
topK_selector.register_recoder(recorder)

model_builder = resnet18
# 选择网络
if args.model == "resnet18":
    model_builder = resnet18
elif args.model == "resnet34":
    model_builder = resnet34
elif args.model == "resnet50":
    model_builder = resnet50
elif args.model == "resnet101":
    model_builder = resnet101
elif args.model == "resnet152":
    model_builder = resnet152
elif args.model == "BaseConvNet":
    model_builder = BaseConvNet

# 7、初始化client, 选择搭载模型等
print("初始化clients")
for c_id in range(client_num_in_total):
    # "ResNet18_GN"
    model = model_builder(num_classes=10)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.wd, amsgrad=True)
    c = Client(model, c_id, args, train_loader[c_id], validation_loader[c_id], test_loader[c_id])
    # 方便更换策略
    c.register(topK_selector=topK_selector, recorder=recorder, broadcaster=broadcaster)
    client_dic[c_id] = c
print("初始化clients完毕")

broadcaster.initialize()

name = None if args.name == '' else args.name

wandb.init(project="dfl2",
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
    print("-----第{}轮训练结束-----".format(rounds))
wandb.finish()

# 9、保存模型
client_model_dic = {}
for id in client_dic:
    client = client_dic[id]
    client_model_dic[id] = client.model
torch.save(client_model_dic, "./model/{:s}_client_dic".format(name))
