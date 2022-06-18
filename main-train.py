from trainer import train
from tester import test
from dataload import DataLoader

import torch, argparse, time, os, sys, yaml
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help="folder for test results")
    parser.add_argument('-batch_size', default=4096, type=int, help="batch size")
    parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_power', default=0.0, type=float, help="power for neg sampling disribution")
    parser.add_argument('-sample_type', default='single', type=str, help="single or double (double treats head and tail dists differently)")

    parser.add_argument('-epochs', default=20, type=int, help="number of epochs")
    parser.add_argument('-save_each', default=1, type=int, help="validate every k epochs")
    parser.add_argument('-emb_dim', default=64, type=int, help="embedding dimension")
    parser.add_argument('-dataset', default='ML_FB', type=str, help="dataset name")

    parser.add_argument('-lr', default=1, type=float, help="learning rate")

    parser.add_argument('-reg_lambda', default=1e-3, type=float, help="kg loss reg term")
    parser.add_argument('-kg_lambda', default=1, type=float, help="l2 regularization parameter")

    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")
    parser.add_argument('-stop_width', default=5, type=int, help="number of SAVES where test is worse for early stopping")

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

if __name__ == '__main__':
    args = get_args()
    assert args.sample_type == 'single' or args.sample_type == 'double'
    if args.save_each is None:
        args.save_each = args.epochs

    # ensure test_name exits, make test result folder
    if args.test_name == None:
        raise ValueError('enter a test name folder using -test_name')
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join("results", args.test_name)
    os.makedirs(save_path) # TODO: update this is continue training...
    save_hyperparams(save_path, args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # print important hyperparameters
    print('epochs: {}, batch size: {}, reg kg: {}, reg lambda: {}, device: {}'. format(
          args.epochs, args.batch_size, args.kg_lambda, args.reg_lambda, device))

    dataloader = DataLoader(args)
    
    # this trains and tests
    print('training')
    train(dataloader, args, device)

