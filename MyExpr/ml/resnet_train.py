import logging
import torch
import sys
import os
from tqdm import tqdm

# 添加环境
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../MyExpr")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

print(sys.path)

# 查看GPU
print(torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print("GPU[{:d}]: {:s}".format(i, torch.cuda.get_device_name(i)))

# 选择GPU
# os.environ['CUDA_VISIBLE_DEVICES'] =  "0"
# print(torch.cuda.device_count())

from MyExpr.dfl.Args import add_args
from MyExpr.data import Dataset
from torch.utils.data import DataLoader

parser = add_args()
# args = parser.parse_args()
args = parser.parse_known_args()[0]

train_set, test_set = Dataset(args)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

from MyExpr.dfl.model.resnet import resnet18
from MyExpr.dfl.model.resnet import resnet34
from MyExpr.dfl.model.resnet import resnet50
from MyExpr.dfl.model.resnet import resnet101
from MyExpr.dfl.model.resnet import resnet152

import torch.nn as nn

# 选择网络模型
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

criterion_CE = nn.CrossEntropyLoss()

import wandb
import time


# 训练函数
def run(model):
    # name = "{:s}-lr{:3f}-bs{:d}".format(args.model, args.lr, args.batch_size)
    print(args.name)

    wandb.init(project="classic-ml",
               entity="kyriegyj",
               name=args.name,
               config=args)

    total_train_iteration = 0

    model.to(args.device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    for epoch in range(args.epochs):
        # train
        start_time = time.perf_counter()
        total_loss = 0
        total_correct = 0
        for iteration, (train_X, train_Y) in enumerate(train_loader):
            optimizer.zero_grad()
            train_X, train_Y = train_X.to(args.device), train_Y.to(args.device)
            outputs = model(train_X)
            loss = criterion_CE(outputs, train_Y)

            pred = outputs.argmax(dim=1)
            correct = pred.eq(train_Y.view_as(pred)).sum()

            loss.backward()
            optimizer.step()

            if "cuda" in args.device:
                loss = loss.cpu()

            loss = loss.detach().numpy()
            acc = (correct / args.batch_size)
            # wandb.log(step=total_train_iteration, data={"loss":loss, "acc:":acc})

            total_loss += loss
            total_correct += correct
            total_train_iteration += 1

        total_acc = (total_correct / (len(train_loader) * args.batch_size))
        wandb.log(step=epoch, data={"total_loss": total_loss, "total_acc": total_acc})
        end_time = time.perf_counter()
        print("epoch[{:d}] spends {:f}s".format(epoch, (end_time - start_time)))

        # test
        total_test_loss = 0
        total_test_correct = 0
        with torch.no_grad():
            for iteration, (test_X, test_Y) in enumerate(test_loader):
                test_X, test_Y = test_X.to(args.device), test_Y.to(args.device)
                outputs = model(test_X)
                loss = criterion_CE(outputs, test_Y)
                pred = outputs.argmax(dim=1)

                if "cuda" in args.device:
                    loss = loss.cpu()

                loss = loss.detach().numpy()
                correct = pred.eq(test_Y.view_as(pred)).sum()

                total_test_loss += loss
                total_test_correct += correct

            total_test_acc = (total_test_correct / (len(test_loader) * args.batch_size))
            wandb.log(step=epoch, data={"total_test_loss": total_test_loss, "total_test_acc": total_test_acc})

    wandb.finish()


model = model_builder(num_classes=10)

run(model)
torch.save(model, "./model/cifar10/"+args.model)