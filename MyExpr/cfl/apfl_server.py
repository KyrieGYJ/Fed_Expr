import copy
import numpy as np
import time
from tqdm import tqdm
import wandb
import torch


# todo 试验
from MyExpr.dfl.model.model_builder import get_model_builder


class APFLServer(object):

    def __init__(self, args):
        np.random.seed(int(round(time.time())))
        self.alpha = 0.5
        self.client_dict = None
        self.client_personal_model_dict = {} # non-invasive
        self.global_model = None
        self.args = args
        self.recorder = None
        self.client_dict = None

    def register_recorder(self, recorder):
        self.recorder = recorder
        self.client_dict = recorder.client_dict

    # assign personal model
    def initialize(self):
        self.global_model = get_model_builder(self.args)(num_classes=10)
        self.broadcast_global_model()
        for i in range(self.args.client_num_in_total):
            personal_model = copy.deepcopy(self.client_dict[i].model.cpu())
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, personal_model.parameters()), lr=self.args.lr)
            self.client_personal_model_dict[i] = {"model":personal_model, "optimizer":optimizer}

    def train(self):
        # randomly select K client participating federated learning
        K = min(int(self.args.broadcast_K * self.args.client_num_in_total), self.args.client_num_in_total)
        selected_clients = np.random.choice(range(self.args.client_num_in_total), K, replace=False)
        print(f"selected_clients: {selected_clients}")
        self.local(selected_clients)
        self.aggregate(selected_clients)
        self.broadcast_global_model()

    # the same as FedAvg
    def aggregate(self, selected_clients):
        print("fedprox aggregate...")
        total_sample_num = 0
        for c_id in selected_clients:
            total_sample_num += len(self.client_dict[c_id].train_set)

        global_model = copy.deepcopy(self.client_dict[0].model.cpu())
        state_dict = global_model.state_dict()

        for key in state_dict: # clear
            state_dict[key].mul_(0.)

        for selected_id in tqdm(selected_clients, desc="aggregating"):
            selected_model = self.client_dict[selected_id].model.cpu()
            # official code uses fair averaging instead
            selected_weight = len(self.client_dict[selected_id].train_set) / total_sample_num
            for key in state_dict:
                temp = selected_model.state_dict()[key].data.mul(selected_weight)
                state_dict[key].add_(temp)

        global_model.load_state_dict(state_dict)
        self.global_model = global_model


    def broadcast_global_model(self):
        for c_id in range(self.args.client_num_in_total):
            self.client_dict[c_id].model.load_state_dict(self.global_model.state_dict()) # internal deep copy
            self.client_dict[c_id].model.to(self.args.device)

    # just add proximal term
    def local(self, selected_clients, turn_on_wandb=True):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0.0, 0.0
        total_epsilon, total_alpha = 0.0, 0.0
        total_num = 0

        for c_id in tqdm(selected_clients, desc="local train"):
            if self.args.enable_dp:
                loss, correct, epsilon, alpha = self.local_train(self.client_dict[c_id])
                total_epsilon += epsilon
                total_alpha += alpha
            else:
                loss, correct = self.local_train(self.client_dict[c_id])
            total_loss += loss
            total_correct += correct
            total_num += len(self.client_dict[c_id].train_set)

        local_train_acc = total_correct / total_num
        avg_local_train_epsilon, avg_local_train_alpha = total_epsilon / len(selected_clients), total_alpha / len(selected_clients)

        if self.args.enable_dp:
            tqdm.write(f"avg_local_train_epsilon:{avg_local_train_epsilon}, avg_local_train_alpha:{avg_local_train_alpha}")

        tqdm.write("local_train_loss:{}, local_train_acc:{}".
              format(total_loss, local_train_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb and turn_on_wandb:
            wandb.log(step=rounds, data={"local_train/loss": total_loss, "local_train/acc": local_train_acc})
            if self.args.enable_dp:
                wandb.log(step=rounds, data={"avg_local_train_epsilon": avg_local_train_epsilon, "avg_local_train_alpha":avg_local_train_alpha})

    # train model and personal_model
    def local_train(self, client):
        optimizer_personal = self.client_personal_model_dict[client.client_id]["optimizer"]
        model_personal = self.client_personal_model_dict[client.client_id]["model"]
        model_personal.train()
        model_personal.to(client.device)
        client.model.train()
        client.model.to(client.device)
        epochs = self.args.epochs
        total_loss, total_correct = 0.0, 0.0
        for epoch in range(epochs):
            for idx, (train_X, train_Y) in enumerate(client.train_loader):
                train_X, train_Y = train_X.to(client.device), train_Y.to(client.device)
                # w_t = w_(t-1) - lr * delta(w_(t-1))
                client.optimizer.zero_grad()
                outputs = client.model(train_X)
                loss = client.criterion_CE(outputs, train_Y)
                pred = outputs.argmax(dim=1)
                correct = pred.eq(train_Y.view_as(pred)).sum()
                loss.backward()
                client.optimizer.step()

                # v_t = v_(t-1) - lr * delta(v_overline_(t-1))
                client.optimizer.zero_grad()
                optimizer_personal.zero_grad()
                outputs_vt = model_personal(train_X)
                outputs_wt = client.model(train_X)
                # https://github.com/MLOPTPSU/FedTorch/issues/6
                outputs_v_overline = self.alpha * outputs_vt + (1 - self.alpha) * outputs_wt
                loss_v_overline = client.criterion_CE(outputs_v_overline, train_Y)
                loss_v_overline.backward()
                optimizer_personal.step()

                pred_v_overline = outputs_v_overline.argmax(dim=1)
                correct_v_overline = pred.eq(train_Y.view_as(pred_v_overline)).sum()

                # adaptive alpha update
                grad_alpha = 0
                for l_params, p_params in zip(client.model.parameters(), model_personal.parameters()):
                    dif = p_params.data - l_params.data
                    grad = self.alpha * p_params.grad.data + (1 - self.alpha) * l_params.grad.data
                    grad_alpha += dif.view(-1).T.dot(grad.view(-1))

                grad_alpha += 0.02 * self.alpha # ???
                alpha_n = self.alpha - self.args.lr * grad_alpha
                self.alpa = np.clip(alpha_n.item(), 0.0, 1.0)
                # global average ignored

                # record train_loss and train_acc of w_t
                # total_loss += loss.cpu().detach().numpy()
                # total_correct += correct

                # record train_loss and train_acc of v_(t-1)_overline
                total_loss += loss_v_overline.cpu().detach().numpy()
                total_correct += correct_v_overline
        total_loss /= epochs
        total_correct /= epochs
        if self.args.enable_dp:
            epsilon, best_alpha = client.privacy_engine.accountant.get_privacy_spent(
                delta=self.args.delta
            )
            print(
                f"Client: {client.client_id} \t"
                f"Loss: {np.mean(total_loss):.6f} "
                f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
            )
            return total_loss, total_correct, epsilon, best_alpha
        return total_loss, total_correct

    def local_test(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0., 0.
        total_num = 0

        for c_id in tqdm(self.client_dict, desc="local_test"):
            client = self.client_dict[c_id]
            loss, correct = client.local_test()
            total_loss += loss
            total_correct += correct
            # print("client {} contains {} test data".format(c_id, len(client.test_loader)))
            total_num += len(client.test_set)

        avg_acc = total_correct / total_num
        print("local_test_loss:{}, avg_local_test_acc:{}".format(total_loss, avg_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"local_test/loss": total_loss, "local_test/avg_acc": avg_acc})
            if avg_acc > self.best_accuracy:
                wandb.run.summary["best_accuracy"] = avg_acc
                self.best_accuracy = avg_acc

    # test for v_t
    def local_personal_test(self):
        rounds = self.recorder.rounds
        total_loss, total_correct = 0., 0.
        total_num = 0

        for c_id in tqdm(self.client_dict, desc="local_test"):
            client = self.client_dict[c_id]
            loss, correct = self.client_local_personal_test(client)
            total_loss += loss
            total_correct += correct
            # print("client {} contains {} test data".format(c_id, len(client.test_loader)))
            total_num += len(client.test_set)

        avg_acc = total_correct / total_num
        print("local_test_loss:{}, avg_local_test_acc:{}".format(total_loss, avg_acc))

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"local_test/loss": total_loss, "local_test/avg_acc": avg_acc})
            if avg_acc > self.best_accuracy:
                wandb.run.summary["best_accuracy"] = avg_acc
                self.best_accuracy = avg_acc

    # can't find a better name for total_loss in a client's test
    def client_local_personal_test(self, client):
        personal_model = self.client_personal_model_dict[client.client_id]["model"]

        total_loss, total_correct = 0., 0
        personal_model.eval()
        personal_model.to(self.args.device)
        with torch.no_grad():
            for _, (test_X, test_Y) in enumerate(client.test_loader):
                test_X, test_Y = test_X.to(self.args.device), test_Y.to(self.args.device)
                outputs = personal_model(test_X)
                loss = client.criterion_CE(outputs, test_Y)

                pred = outputs.argmax(dim=1)
                correct = pred.eq(test_Y.view_as(pred)).sum()
                total_loss += loss.cpu().detach().numpy()
                total_correct += correct
        return total_loss, total_correct