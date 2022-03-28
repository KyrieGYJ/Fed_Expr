from MyExpr.dfl.model.resnet import resnet18
from MyExpr.dfl.model.resnet import resnet34
from MyExpr.dfl.model.resnet import resnet50
from MyExpr.dfl.model.resnet import resnet101
from MyExpr.dfl.model.resnet import resnet152
from MyExpr.dfl.model.cnn import BaseConvNet
from MyExpr.dfl.model.cnn import FedAvgCNN

def get_model_builder(args):
    model_builder = resnet18
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
    elif args.model == "FedAvgCNN":
        model_builder = FedAvgCNN
    return model_builder