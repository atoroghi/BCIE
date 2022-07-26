import os, re, yaml, argparse, torch, math, sys
import numpy as np
import matplotlib.pyplot as plt
from tester import test
from trainer import train
from dataload import DataLoader

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev')
    parser.add_argument('-load_epoch', default=0)
    return parser.parse_args() 

if __name__ == '__main__':
    tune_name = 'gauss'
    folds = 5
    
    # search through all folders
    for i in range(folds):
        path = 'results/{}/fold_{}'.format(tune_name, i)
        folders = os.listdir(path)
        folders = [f for f in folders if 'train' in f]
        folders = sorted(folders, key=natural_key)

        # get best performance and hps
        perf, arg_perf = [], []
        for f in folders:
            try:
                scores = np.load(os.path.join(path, f, 'stop_metric.npy'), allow_pickle=True)
                perf.append(np.max(scores))
                arg_perf.append(np.argmax(scores))
            except:
                print('skipped: ', f)

        best_run = np.argmax(perf)
        best_score = np.max(perf)
        best_arg = arg_perf[np.argmax(perf)]
        print('best score: {}, best folder: {}, best epoch: {}'.format(best_score, best_run, best_arg))
        continue

        ######################################
        # test best run from given fold
        args = get_args()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # get args from file
        path = os.path.join('results', tune_name, 'fold_{}'.format(i), 'train_{}'.format(best_run))
        with open(os.path.join(path, 'info.yml'), 'r') as f:
            yml = yaml.safe_load(f)
            for key in yml.keys():
                setattr(args, key, yml[key])
        
        dataloader = DataLoader(args)

        # load model
        load_path = os.path.join(path, 'models', 'epoch_{}.chkpnt'.format(best_arg))
        model = torch.load(load_path).to(device)
        test(model, dataloader, args.load_epoch, args, 'test')
