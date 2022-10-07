import torch
from launch import get_args
from trainer import train
from tester import test
from dataload import DataLoader
import wandb
import argparse
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', default=0.1, type=float, help="folder for test results")
    parser.add_argument('--model_type', default='Simple', type=str, help="model type (svd, Simple, etc)")

    # hyper-parameters (optimized)
    parser.add_argument('--lr', default=1, type=float, help="learning rate")
    parser.add_argument('--batch_size', default=4096, type=int, help="batch size")
    parser.add_argument('--emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('--reg_lambda', default=1e-3, type=float, help="kg loss reg term")
    parser.add_argument('--kg_lambda', default=1, type=float, help="l2 regularization parameter")   
    parser.add_argument('--neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    parser.add_argument('--neg_power', default=0.0, type=float, help="power for neg sampling disribution")
    parser.add_argument('--init_scale', default=None, type=float, help="std for normal, gain for uniform")
    parser.add_argument('--hinge_margin', default=1, type=float, help="in case of margin loss, margin")
    
    # other hyper-params
    parser.add_argument('--reg_type', default='gauss', type=str, help="tilt or gauss")
    parser.add_argument('--loss_type', default='softplus', type=str, help="softplus or gauss")
    parser.add_argument('--reduce_type', default='sum', type=str, help="sum or mean")
    parser.add_argument('--optim_type', default='adagrad', type=str, help="adagrad or adam")
    parser.add_argument('--sample_type', default='double', type=str, help="single or double (double treats head and tail dists differently)")
    parser.add_argument('--init_type', default='uniform', type=str, help="uniform or normal")
    parser.add_argument('--kg', default='kg', type=str, help="kg or no_kg")
    parser.add_argument('--type_checking',default='no', type=str, help="doing type checking or not")

    # optimization and saving
    parser.add_argument('--epochs', default=20, type=int, help="number of epochs")
    parser.add_argument('--save_each', default=1, type=int, help="validate every k epochs")
    parser.add_argument('--dataset', default='ML_FB', type=str, help="dataset name")
    parser.add_argument('--stop_width', default=4, type=int, help="number of SAVES where test is worse for early stopping")
    parser.add_argument('--fold', default=0, type=int, help="fold to use data from")

    args = parser.parse_args()
    return args


def config_to_args(config):
    if isinstance(config, dict):
        args = argparse.Namespace(**config)
    else:
        raise TypeError(f"expecting type dict or argparser.Namespace, got {type(config)}")
    return args

def train_func(config):
    os.chdir('/home/admin/Desktop/BK-KGE')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #args_new = config_to_args(config)
    args_new = get_args()
    dataloader = DataLoader(args_new)
    hits10 = train(dataloader, args_new, device)
    wandb.log({"hits10":hits10})


if __name__ == '__main__':
    wandb.login(key="d606ae06659873794e6a1a5fb4f55ffc72aac5a1")
    wandb.config = {
    "lr": 0.001,
    "epochs": 10,
    "batch_size": 4096
    }
    wandb.init(project="SimplE", entity="atoroghi")
    os.environ['WANDB_API_KEY']='d606ae06659873794e6a1a5fb4f55ffc72aac5a1'
    os.environ['WANDB_USERNAME']='atoroghi'
    config = wandb.config
    train_func(config)
    #wandb.watch(train_func)







# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-test_name', default='dev', type=str, help="folder for test results")
#     parser.add_argument('-model_type', default='Simple', type=str, help="model type (svd, Simple, etc)")

#     # hyper-parameters (optimized)
#     parser.add_argument('-lr', default=1, type=float, help="learning rate")
#     parser.add_argument('-batch_size', default=4096, type=int, help="batch size")
#     parser.add_argument('-emb_dim', default=128, type=int, help="embedding dimension")
#     parser.add_argument('-reg_lambda', default=1e-3, type=float, help="kg loss reg term")
#     parser.add_argument('-kg_lambda', default=1, type=float, help="l2 regularization parameter")   
#     parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
#     parser.add_argument('-neg_power', default=0.0, type=float, help="power for neg sampling disribution")
#     parser.add_argument('-init_scale', default=None, type=float, help="std for normal, gain for uniform")
#     parser.add_argument('-hinge_margin', default=1, type=float, help="in case of margin loss, margin")
    
#     # other hyper-params
#     parser.add_argument('-reg_type', default='tilt', type=str, help="tilt or gauss")
#     parser.add_argument('-loss_type', default='gauss', type=str, help="softplus or gauss")
#     parser.add_argument('-reduce_type', default='mean', type=str, help="sum or mean")
#     parser.add_argument('-optim_type', default='adam', type=str, help="adagrad or adam")
#     parser.add_argument('-sample_type', default='double', type=str, help="single or double (double treats head and tail dists differently)")
#     parser.add_argument('-init_type', default='uniform', type=str, help="uniform or normal")
#     parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")

#     # optimization and saving
#     parser.add_argument('-epochs', default=20, type=int, help="number of epochs")
#     parser.add_argument('-save_each', default=1, type=int, help="validate every k epochs")
#     parser.add_argument('-dataset', default='ML_FB', type=str, help="dataset name")
#     parser.add_argument('-stop_width', default=4, type=int, help="number of SAVES where test is worse for early stopping")

#     args = parser.parse_args()
#     return args, argparse.Namespace()

# def config_to_args(config, ns):
#     if isinstance(config, dict):
#         for k , v in config.items():
#             setattr(ns, k , v)
#     #args = argparse.Namespace(**config)
#     else:
#         raise TypeError(f"expecting type dict or argparser.Namespace, got {type(config)}")
#     return ns

# def train_func(config):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     args = get_args()
#     args_new = config_to_args(config, args)
#     dataloader = DataLoader(args_new)
#     hits10 = train(dataloader, args_new, device)

# def main():  

#     result = tune.run(
#     tune.with_parameters(train_func),
#     resources_per_trial={"cpu": 2, "gpu": 1},
#     config = {
#         "emb_dim" : tune.choice([32,64]),
#         #"lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size" : tune.choice([64]),
#         "reg_lambda" : tune.choice([0.1]),
#         "neg_ratio" : tune.choice([10])
#     },
#     metric = "hits10",
#     mode = "max",
#     search_alg = "optuna"
#     )
#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))

#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["hits10"]))

# if __name__ == '__main__':
#     main()