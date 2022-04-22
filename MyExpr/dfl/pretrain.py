import torch
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../MyExpr")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from MyExpr.dfl.args import add_args
from MyExpr.dfl.main import initialize

parser = add_args()
args = parser.parse_args()
args.turn_on_wandb = False
if "non-iid" in args.data_distribution:
    # 优先按比例划分
    args.num_clients_per_dist = int(args.client_num_in_total / args.num_distributions)

name = None if args.name == '' else args.name
project_name = None if args.project_name == '' else args.project_name

# local train before federation #
if args.pretrain_epoch > 0:
    model_dict_fname = f"./precomputed/{project_name}/{args.model}_c{args.client_num_in_total}" \
                       f"_{args.data_distribution}_dn{args.num_distributions}_pe{args.pretrain_epoch}"
    model_dict = {i: None for i in range(args.client_num_in_total)}
    try:
        model_dict = torch.load(model_dict_fname)
        print(f'##### model dictionary {model_dict_fname} exists, no need to compute')
    except:
        print(f'##### no model dictionary was found, newly train {model_dict_fname}')
        print(f"当前pid: {os.getpid()}")
        if not os.path.exists(f'./precomputed/{project_name}'):
            os.makedirs(f'./precomputed/{project_name}')
        client_dict, data, _, _, _ = initialize(args)
        pretrain_epoch = args.pretrain_epoch

        for i in tqdm(model_dict, desc="pretrain each client"):
            model = client_dict[i].model
            model.train()
            model.to(args.device)
            total_loss, total_correct = 0.0, 0.0
            for e in tqdm(range(pretrain_epoch), desc="current client"):
                for idx, (train_X, train_Y) in enumerate(client_dict[i].train_loader):
                    train_loss, correct = client_dict[i].train(train_X, train_Y)
                    total_loss += train_loss
                    total_correct += correct
                # print("client {}: local train takes {} iteration".format(self.client_id, iteration))
            total_loss /= pretrain_epoch
            total_correct /= pretrain_epoch
            print(f"client {i} train_loss: {total_loss}, train_acc: {total_correct/len(client_dict[i].train_set)}")

        for i in range(args.client_num_in_total):
            model_dict[i] = client_dict[i].model.cpu().state_dict()

        fname = f"{args.model}_c{args.client_num_in_total}" \
                f"_{args.data_distribution}_dn{args.num_distributions}_pe{args.pretrain_epoch}"
        torch.save(model_dict, f"./precomputed/{project_name}/{fname}")

