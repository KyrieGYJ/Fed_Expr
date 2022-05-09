# import numpy as np
# import time
# from tqdm import tqdm
# import copy
# import torch
# import heapq
#
# # deprecated
# class PENSTrainer(object):
#
#     def __init__(self, args):
#         self.args = args
#         self.recorder = None
#         self.client_dict = None
#         self.broadcaster = None
#         self.step_one_rounds = 200
#         self.step_two_rounds = 333
#         self.n_sampled = 10
#         self.n_peer = 20
#         self.m = 2
#
#     def register_recorder(self, recorder):
#         self.recorder = recorder
#         self.client_dict = recorder.client_dict
#         self.broadcaster = recorder.broadcaster
#         self.dist_client_dict = recorder.data.dist_client_dict
#         self.client_dist_dict = recorder.data.client_dist_dict
#         self.test_loader = recorder.data.test_all
#         self.test_data = recorder.data.test_data
#
#     def initialize(self):
#         for c_id in self.client_dict:
#             self.client_dict[c_id].register_response_hook(self.on_received1)
#
#     def randomly_communicate(self):
#         self.broadcaster.send = self.broadcaster.random # singleton
#         for c_id in tqdm(self.client_dict, desc="benign broadcast"):
#             self.client_dict[c_id].broadcast()
#
#     def train(self):
#
#         # random
#         # select
#         # gossip
#         pass
#
#     def gossip_learning(self):
#
#
#
#     def on_received1(self, client, sender_id):
#         # need to deep copy
#         received_model = self.client_dict[sender_id].model
#         validation_loss = self.calc_validation_loss(client, received_model)
#         received_model = copy.deepcopy(received_model.cpu())
#         client.received_model_dict[received_model] = {"model":received_model, "loss":validation_loss} # non invasive for convenient
#         if len(received_model) >= self.n_sampled:
#             top_k = heapq.nlargest(self.m, client.received_model_dict, lambda x: -x["loss"])
#             self.model_average(client, top_k)
#             client.received_model_dict = {}
#
#     def model_average(self, host, candidate):
#         n = len(candidate) + 1
#         state_dict = host.model.state_dict()
#
#         for key in state_dict:
#             state_dict[key].div_(n)
#
#         for model_and_loss in candidate:
#             neighbor_model = model_and_loss["model"]
#             for key1, key2 in zip(state_dict, neighbor_model.state_dict()):
#                 temp = neighbor_model.state_dict()[key2].data.div(n)
#                 state_dict[key1].add_(temp)
#
#     def calc_validation_loss(self, host, received_model):
#         received_model.eval()
#         received_model.to(self.args.device)
#         loss = 0.
#         with torch.no_grad():
#             for data, label in host.validation_loader:
#                 data, label = data.to(self.args.device), label.to(self.args.device)
#                 received_outputs = received_model(data)
#                 received_loss = host.criterion_CE(received_outputs, label)
#                 loss += received_loss.cpu().detach().numpy()
#         return loss