import copy
import numpy as np
import time
from tqdm import tqdm
import wandb
import torch
from MyExpr.dfl.model.model_builder import get_model_builder


# todo 实验
class pFedMeServer(object):
    # server as trainer
    def __init__(self, args):
        np.random.seed(int(round(time.time())))
        self.beta = 1
        self.lamda = 15
        # self.personal_learning_rate = 0.1 # lr for personalized step
        self.learning_rate = args.lr
        self.client_dict = None
        self.global_model = None
        self.args = args
        self.recorder = None
        self.client_dict = None
        self.best_accuracy = 0.

    def register_recorder(self, recorder):
        self.recorder = recorder
        self.client_dict = recorder.client_dict

    def initialize(self):
        self.global_model = get_model_builder(self.args)(num_classes=10)
        self.broadcast_global_model()

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
            selected_weight = len(self.client_dict[selected_id].train_set) / total_sample_num
            for key in state_dict:
                temp = selected_model.state_dict()[key].data.mul(selected_weight)
                state_dict[key].add_(temp)

        global_model.load_state_dict(state_dict)
        for pre_param, param in zip(self.global_model.parameters(), global_model.parameters()):
            pre_param.data = (1 - self.beta) * pre_param.data + self.beta * param.data

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
        self.recorder.record_global_history("train_loss", total_loss)
        self.recorder.record_global_history("train_acc", local_train_acc)

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb and turn_on_wandb:
            wandb.log(step=rounds, data={"local_train/loss": total_loss, "local_train/acc": local_train_acc})
            if self.args.enable_dp:
                wandb.log(step=rounds, data={"avg_local_train_epsilon": avg_local_train_epsilon, "avg_local_train_alpha":avg_local_train_alpha})

    def local_train(self, client):
        model_t = copy.deepcopy(client.model.cpu()) # save w_t_r
        client.model.train()
        client.model.to(self.args.device)
        epochs = self.args.epochs
        total_loss, total_correct = 0.0, 0.0

        # finding aproximate theta for personalization -> personalize(w_t_r)
        for epoch in range(epochs):
            for idx, (train_X, train_Y) in enumerate(client.train_loader):
                client.optimizer.zero_grad()
                train_X, train_Y = train_X.to(client.device), train_Y.to(client.device)
                outputs = client.model(train_X)
                loss = client.criterion_CE(outputs, train_Y)
                pred = outputs.argmax(dim=1)
                correct = pred.eq(train_Y.view_as(pred)).sum()
                loss.backward()
                client.optimizer.step()
                total_loss += loss.cpu().detach().numpy()
                total_correct += correct


        # update local weight after finding aproximate theta
        # w_t_(r+1) = w_t_r - lr * lamda * (w_t_r - personalize(w_t_r))
        client.model.cpu()
        for new_param, localweight in zip(client.model.parameters(), model_t.parameters()):
            localweight.data = localweight.data - self.lamda * self.learning_rate * (
                        localweight.data - new_param.data)
        client.model.load_state_dict(model_t.state_dict())
        del model_t


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
        self.recorder.record_global_history("test_loss", total_loss)
        self.recorder.record_global_history("test_acc", avg_acc)

        # print("-----上传至wandb-----")
        if self.args.turn_on_wandb:
            wandb.log(step=rounds, data={"local_test/loss": total_loss, "local_test/avg_acc": avg_acc})
            if avg_acc > self.best_accuracy:
                wandb.run.summary["best_accuracy"] = avg_acc
                self.best_accuracy = avg_acc
        if avg_acc > self.best_accuracy:
            self.recorder.history.best_accuracy = avg_acc