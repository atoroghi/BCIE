
from trainer import train
from tester import test
from dataload import DataLoader

import torch, argparse, time, os, sys, yaml, math
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default=0.1, type=float, help="folder for test results")
    parser.add_argument('-model_type', default='simple', type=str, help="model type (svd, Simple, etc)")

    # hyper-parameters (optimized)
    parser.add_argument('-lr', default=1, type=float, help="learning rate")
    parser.add_argument('-batch_size', default=4096, type=int, help="batch size")
    parser.add_argument('-emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('-reg_lambda', default=1e-3, type=float, help="kg loss reg term")
    parser.add_argument('-kg_lambda', default=1, type=float, help="l2 regularization parameter")   
    parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_power', default=0.0, type=float, help="power for neg sampling disribution")
    parser.add_argument('-init_scale', default=1, type=float, help="std for normal, gain for uniform")
    parser.add_argument('-hinge_margin', default=1, type=float, help="in case of margin loss, margin")
    
    # for svd
    parser.add_argument('-rank', default=10, type=int, help="rank for svd")
    parser.add_argument('-n_iter', default=5, type=int, help="number of iterations for approx method")
    
    # other hyper-params
    parser.add_argument('-reg_type', default='tilt', type=str, help="tilt or gauss")
    parser.add_argument('-loss_type', default='gauss', type=str, help="softplus or gauss")
    parser.add_argument('-reduce_type', default='mean', type=str, help="sum or mean")
    parser.add_argument('-optim_type', default='adam', type=str, help="adagrad or adam")
    parser.add_argument('-sample_type', default='double', type=str, help="single or double (double treats head and tail dists differently)")
    parser.add_argument('-init_type', default='uniform', type=str, help="uniform or normal")
    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")
    parser.add_argument('-type_checking',default='no', type=str, help="doing type checking or not")

    # optimization, saving and data
    parser.add_argument('-epochs', default=30, type=int, help="number of epochs")
    parser.add_argument('-save_each', default=1, type=int, help="validate every k epochs")
    parser.add_argument('-dataset', default='ML_FB', type=str, help="dataset name")
    parser.add_argument('-stop_width', default=4, type=int, help="number of SAVES where test is worse for early stopping")
    parser.add_argument('-fold', default=0, type=int, help="fold to use data from")

    args = parser.parse_args()
    return args

def save_hyperparams(path, args):
    dct={}
    for key in args.__dict__:
        val = getattr(args, key)
        dct[key] = val
    
    with open(os.path.join(path, 'info.yml'), 'w') as f:
            yaml.dump(dct, f, sort_keys=False,
                      default_flow_style=False)

def main(args):
    assert args.sample_type in ['single', 'double']
    assert args.reg_type in ['gauss', 'tilt', 'tilt']
    assert args.loss_type in ['softplus', 'gauss', 'hinge', 'PSL']
    assert args.optim_type in ['adagrad', 'adam']
    assert args.init_type in ['uniform', 'normal']

    if args.init_scale == 1: 
        print('manual init scale')
        args.init_scale = 6.0 / math.sqrt(args.emb_dim)

    if args.save_each is None:
        args.save_each = args.epochs

    # ensure test_name exits, make test result folder
    if args.test_name == None:
        raise ValueError('enter a test name folder using -test_name')
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join("results", str(args.test_name))
    os.makedirs(save_path, exist_ok=True) # TODO: update this is continue training...
    save_hyperparams(save_path, args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # print important hyperparameters
    #print('fold {}, device: {}'. format(args.fold, device))

    dataloader = DataLoader(args)
    
    # this trains and tests
    #print('training')
    hits10 = train(dataloader, args, device)
    print(hits10)

if __name__ == '__main__':
    args = get_args()
    main(args)