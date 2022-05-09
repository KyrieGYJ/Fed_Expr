import copy
import numpy as np
import time
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch
import torch.nn as nn

from MyExpr.dfl.model.model_builder import get_model_builder, get_optimizer, get_lr_scheduler

# todo bugged
class FedEMServer(object):
    # server as trainer
    def __init__(self, args):
        np.random.seed(int(round(time.time())))
        self.global_learners_ensemble = None
        self.client_learners_ensemble_dict = {}
        self.clients_weights = None

        self.n_learners = 3
        self.seed = 0
        self.client_dict = None
        self.global_model = None
        self.args = args
        self.recorder = None
        self.client_dict = None

    def register_recorder(self, recorder):
        self.recorder = recorder
        self.client_dict = recorder.client_dict

    def initialize(self):
        self.global_learners_ensemble = self.get_learners_ensemble()
        for i in range(self.args.client_num_in_total):
            self.client_learners_ensemble_dict[i] = self.get_learners_ensemble()
        self.broadcast_global_model()

        self.clients_weights = \
            torch.tensor(
                [len(self.client_dict[client_id].train_set) for client_id in self.client_dict],
                dtype=torch.float32
            )
        self.clients_weights = self.clients_weights / self.clients_weights.sum()

    # in official code style
    def train(self, turn_on_wandb=True):
        # randomly select K client participating federated learning
        sampled_clients = self.sample_client()
        # local
        total_loss, total_correct = 0., 0.
        total_num = 0
        for client_id in sampled_clients:
            client = self.client_dict[client_id]
            loss, correct = self.client_step(client)
            total_loss += loss
            total_correct += total_correct
            total_num += len(self.client_dict[client_id].train_set)

        local_train_acc = total_correct / total_num

        rounds = self.recorder.rounds
        tqdm.write("local_train_loss:{}, local_train_acc:{}".
                   format(total_loss, local_train_acc))
        if self.args.turn_on_wandb and turn_on_wandb:
            wandb.log(step=rounds, data={"local_train/loss": total_loss, "local_train/acc": local_train_acc})
        # aggregate
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [self.client_learners_ensemble_dict[client_id][learner_id] for client_id in self.client_dict]
            average_learners(learners, learner, weights=self.clients_weights, args=self.args)

        # broadcast global model
        # assign the updated model to all clients
        self.broadcast_global_model()

    def sample_client(self):
        K = min(int(self.args.broadcast_K * self.args.client_num_in_total), self.args.client_num_in_total)
        selected_clients = np.random.choice(range(self.args.client_num_in_total), K, replace=False)
        print(f"selected_clients: {selected_clients}")
        return selected_clients

    # non-invasive
    def client_step(self, client):
        """
        perform on step for the client
       """
        # self.counter += 1
        samples_weights = self.client_update_sample_weights(client)
        self.client_update_learners_weights(client, samples_weights)

        total_loss, total_correct = self.learners_ensemble_fit_epochs(client, samples_weights)
        return total_loss, total_correct

    def client_update_sample_weights(self, client):
        all_losses = self.learners_ensemble_gather_losses(client)
        # print(f"client {client.client_id} all_losses.shape {all_losses.shape}")
        samples_weights = F.softmax((torch.log(self.client_learners_ensemble_dict[client.client_id].learners_weights) - all_losses.T),
                                         dim=1).T
        # print(f"client {client.client_id} samples_weights.shape {samples_weights.shape}")
        return samples_weights

    def client_update_learners_weights(self, client, samples_weights):
        self.client_learners_ensemble_dict[client.client_id].learners_weights = samples_weights.mean(dim=1)
        print(f"self.client_learners_ensemble_dict[client.client_id].learners_weights: {self.client_learners_ensemble_dict[client.client_id].learners_weights.shape}")

    def learners_ensemble_gather_losses(self, client):
        """
        gathers losses for all sample in iterator for each learner in ensemble
        """
        learner_ensemble = self.client_learners_ensemble_dict[client.client_id]
        n_samples = len(client.validation_loader.dataset)
        all_losses = torch.zeros(len(learner_ensemble.learners), n_samples)
        learner_losses_dict = {}
        for learner_id, _ in enumerate(learner_ensemble.learners):
            learner_losses_dict[learner_id] = []
        with torch.no_grad():
            for idx, (validation_X, validation_Y) in enumerate(client.validation_loader):
                validation_X, validation_Y = validation_X.to(client.device).type(torch.float32), validation_Y.to(client.device)
                for learner_id, learner in enumerate(learner_ensemble.learners):
                    learner_model = learner["model"].to(client.device)
                    learner_model.eval()
                    y_pred = learner_model(validation_X)
                    learner_loss = learner["criterion"](y_pred, validation_Y).squeeze()
                    learner_losses_dict[learner_id].append(learner_loss.cpu().detach())
                    # all_losses[learner_id][idx] = learner_loss.cpu()
        for learner_id, _ in enumerate(learner_ensemble.learners):
            all_losses[learner_id] = torch.cat(learner_losses_dict[learner_id], 0)
        return all_losses

    def learners_ensemble_fit_epochs(self, client, weights=None):
        """
        updates learners using  one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        total_loss, total_correct = 0., 0.
        learner_ensemble = self.client_learners_ensemble_dict[client.client_id]
        for epoch in range(self.args.epochs):
            for idx, (train_X, train_Y) in enumerate(client.train_loader):
                train_X, train_Y = train_X.to(client.device), train_Y.to(client.device)
                # learner.fit_epoch()
                for learner_id, learner in enumerate(learner_ensemble.learners):
                    learner_model = learner["model"].to(client.device)
                    learner_optimizer = learner["optimizer"]
                    learner_criterion = learner["criterion"]
                    # learner_scheduler = learner["scheduler"]
                    learner_model.train()
                    learner_optimizer.zero_grad()
                    y_pred = learner_model(train_X)
                    loss_vec = learner_criterion(y_pred, train_Y)

                    if weights is not None:
                        weights = weights.to(client.device)
                        loss = (loss_vec.T @ weights[idx]) / loss_vec.size(0)
                    else:
                        loss = loss_vec.mean()
                    loss.backward()

                    pred = y_pred.argmax(dim=1)
                    correct = pred.eq(train_Y.view_as(pred)).sum()
                    total_loss += loss.cpu().detach().numpy()
                    total_correct += correct

                    learner_optimizer.step()
                    # if learner_scheduler is not None:
                    #     learner_scheduler.step()
        total_loss /= self.args.epochs * len(learner_ensemble.learners)
        total_correct /= self.args.epochs * len(learner_ensemble.learners)
        return total_loss, total_correct


    def broadcast_global_model(self):
        for client_id in self.client_dict:
            for learner_id, learner in enumerate(self.client_learners_ensemble_dict[client_id].learners):
                learner["model"].load_state_dict(self.global_learners_ensemble.learners[learner_id]["model"].state_dict())

    def get_learners_ensemble(self):
        # n_learners means multi-model
        torch.manual_seed(self.seed)
        learners = []
        for learner_id in range(self.n_learners):
            model = get_model_builder(self.args)(num_classes=10)
            criterion = nn.CrossEntropyLoss(reduction="none")
            optimizer = get_optimizer(self.args.client_optimizer, model, lr_initial=self.args.lr)
            scheduler = get_lr_scheduler(optimizer, self.args.lr_scheduler)
            learner = {"model":model, "criterion":criterion, "optimizer":optimizer, "scheduler":scheduler}
            learners.append(learner)

        learners_weights = torch.ones(self.n_learners) / self.n_learners
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights, args=self.args)


class LearnersEnsemble(object):

    def __init__(self, learners, learners_weights, args):
        self.learners = learners
        self.learners_weights = learners_weights
        self.args = args


def average_learners(
        learners,
        target_learner,
        args,
        weights=None,
        average_params=True,
        average_gradients=False):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(args.device)

    target_state_dict = target_learner["model"].state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner["model"].state_dict(keep_vars=True)

                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        print(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner["model"].state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()