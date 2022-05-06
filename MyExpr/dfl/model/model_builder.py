import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np

from MyExpr.dfl.model.resnet import resnet18
from MyExpr.dfl.model.resnet import resnet34
from MyExpr.dfl.model.resnet import resnet50
from MyExpr.dfl.model.resnet import resnet101
from MyExpr.dfl.model.resnet import resnet152
from MyExpr.dfl.model.cnn import BaseConvNet
from MyExpr.dfl.model.cnn import FedAvgCNN
from MyExpr.dfl.model.cnn import TFConvNet


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
    elif args.model == "TFConvNet":
        model_builder = TFConvNet
    # print(f"采用网络{args.model}进行试验")
    return model_builder


def get_optimizer(optimizer_name, model, lr_initial):
    """
    Gets torch.optim.Optimizer given an optimizer name, a model and learning rate

    :param optimizer_name: possible are adam and sgd
    :type optimizer_name: str
    :param model: model to be optimized
    :type optimizer_name: nn.Module
    :param lr_initial: initial learning used to build the optimizer
    :type lr_initial: float
    :param mu: proximal term weight; default=0.
    :type mu: float
    :return: torch.optim.Optimizer

    """

    if optimizer_name == "adam":
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_initial,
            weight_decay=5e-4
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_initial,
            momentum=0.9,
            weight_decay=5e-4
        )

    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")