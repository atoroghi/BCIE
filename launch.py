from ossaudiodev import SNDCTL_SYNTH_REMOVESAMPLE
from trainer import train
from tester import test
from dataload import DataLoader

import torch, argparse, time, os, sys, yaml
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help="folder for test results")

    # hyper-parameters (optimized)
    parser.add_argument('-lr', default=1, type=float, help="learning rate")
    parser.add_argument('-batch_size', default=4096, type=int, help="batch size")
    parser.add_argument('-emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('-reg_lambda', default=1e-3, type=float, help="kg loss reg term")
    parser.add_argument('-kg_lambda', default=1, type=float, help="l2 regularization parameter")   
    parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_power', default=0.0, type=float, help="power for neg sampling disribution")
    parser.add_argument('-init_scale', default=1.0, type=float, help="std for normal, gain for uniform")
    
    # other hyper-params
    parser.add_argument('-reg_type', default='gauss', type=str, help="tilt or gauss")
    parser.add_argument('-loss_type', default='softplus', type=str, help="softplus or sigmoid")
    parser.add_argument('-optim_type', default='adagrad', type=str, help="adam or sgd")
    parser.add_argument('-sample_type', default='double', type=str, help="single or double (double treats head and tail dists differently)")
    parser.add_argument('-init_type', default='uniform', type=str, help="embedding initialization")
    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")

    # optimization and saving
    parser.add_argument('-epochs', default=20, type=int, help="number of epochs")
    parser.add_argument('-save_each', default=1, type=int, help="validate every k epochs")
    parser.add_argument('-dataset', default='ML_FB', type=str, help="dataset name")
    parser.add_argument('-stop_width', default=4, type=int, help="number of SAVES where test is worse for early stopping")

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
    assert args.reg_type in ['gauss', 'tilt_mean', 'tilt_sum']
    assert args.loss_type in ['softplus', 'gauss']
    assert args.optim_type in ['adagrad', 'adam']
    assert args.init_type in ['uniform', 'normal']

    if args.save_each is None:
        args.save_each = args.epochs

    # ensure test_name exits, make test result folder
    if args.test_name == None:
        raise ValueError('enter a test name folder using -test_name')
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join("results", args.test_name)
    os.makedirs(save_path, exist_ok=True) # TODO: update this is continue training...
    save_hyperparams(save_path, args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # print important hyperparameters
    #print('epochs: {}, batch size: {}, reg kg: {}, reg lambda: {}, device: {}'. format(
    #      args.epochs, args.batch_size, args.kg_lambda, args.reg_lambda, device))

    dataloader = DataLoader(args)
    
    # this trains and tests
    #print('training')
    train(dataloader, args, device)

if __name__ == '__main__':
    args = get_args()
    main(args)

