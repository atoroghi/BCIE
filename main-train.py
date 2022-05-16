from trainer import train
from tester import test
from dataload import LoadDataset
from measure import Measure
from recommender import Recommender
from updater import Updater
import matplotlib.pyplot as plt
import torch, argparse, time, os, sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default=None, type=str, help="folder for test results")
    parser.add_argument('-batch_size', default=16384, type=int, help="batch size")
    parser.add_argument('-neg_ratio', default=10, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_power', default=0, type=float, help="power for neg sampling disribution")

    parser.add_argument('-ne', default=200, type=int, help="number of epochs")
    parser.add_argument('-save_each', default=100, type=int, help="validate every k epochs")
    parser.add_argument('-workers', default=8, type=int, help="threads for dataloader")
    parser.add_argument('-emb_dim', default=64, type=int, help="embedding dimension")

    parser.add_argument('-ni', default=0, type=float, help="noise intensity")
    parser.add_argument('-lr', default=1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.0, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="ML_FB", type=str, help="wordnet dataset")
    parser.add_argument('-max_iters_laplace', default=1000, type=int, help="Maximum number of iterations for Laplace Approximation")
    parser.add_argument('-alpha', default=0.01, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-etta', default=1.0, type=float, help="Learning rate for Laplace Approximation")
    args = parser.parse_args()
    return args

def save_hyperparams(path, args):
    with open(os.path.join(path, 'info.txt'), 'w') as f:
        f.write('batch size: {}\n'.format(args.batch_size))
        f.write('epochs: {}\n'.format(args.ne))
        f.write('learning rate: {}\n'.format(args.lr))
        f.write('lambda regularizer: {}\n'.format(args.reg_lambda))
        f.write('dataset: {}\n'.format(args.dataset))
        f.write('embedding dimension: {}\n'.format(args.emb_dim))
        f.write('negative ratio: {}\n'.format(args.neg_ratio))
        f.write('negative power: {}\n'.format(args.neg_power))
        f.write('alpha: {}\n'.format(args.alpha))
        f.write('etta: {}\n'.format(args.etta))
        f.write('noise intensity: {}\n'.format(args.ni))
        f.write('max laplace iterations: {}\n'.format(args.max_iters_laplace))

if __name__ == '__main__':
    args = get_args()

    # ensure test_name exits, make test result folder
    if args.test_name == None:
        raise ValueError('enter a test name folder using -test_name')
    os.makedirs('results', exist_ok=True)
    save_path = os.path.join("results", args.test_name)
    os.makedirs(save_path, exist_ok=True)
    save_hyperparams(save_path, args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # print important hyperparameters
    print('epochs: {}, batch size: {}, dataset: {}, device: {}'. format(
          args.ne, args.batch_size, args.dataset, device
    ))

    #print('training')
    #dataset = LoadDataset('train', args)
    #train(dataset, args, device)

    print('testing')
    dataset = LoadDataset('test', args)
    model_path = os.path.join('models', args.test_name, str(args.ne) + '.chkpnt')
    test(dataset, args, device)

