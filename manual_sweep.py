import os, re, yaml, argparse, torch, math, sys, subprocess
from random import Random
import numpy as np
import matplotlib.pyplot as plt
from tester import test
from trainer import train
from dataload import DataLoader
from sklearn.ensemble import RandomForestRegressor
from svd import svd
from WRMF_torch import wrmf
from critique import critiquing

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cv_tune_name', default='tuned', type = str)
    parser.add_argument('-folds', default=5, type=int, help='no of folds')
    parser.add_argument('-batch', default=3, type=int, help='no of procs running simultaneously')
    parser.add_argument('-name', default='diff', type = str, help = 'name of the test')

    return parser.parse_args() 

def test_fold(path, tune_name, best_folder, best_epoch):
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model

    #save_path = os.path.join(path, 'z_sweep')
    #os.makedirs(save_path, exist_ok = True)

    best_path = os.path.join(path, best_folder)
    folder_counter = len(os.listdir(path))
    batch_counter = 0
    batch = args.batch
    z_precs = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    z_means = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    with open(os.path.join(best_path, 'crit hps.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        for key in yml.keys():
            #TODO: fix this, while saving the yml file the numerical values shouldn't be strings
            if key in ['session_length', 'multi_k', 'num_users', 'sim_k', 'batch', 'samples', 'folds', 'epochs_all']:
                setattr(args, key, int(yml[key]))
            else:
                try:
                    setattr(args, key, float(yml[key]))
                except:
                    setattr(args, key, yml[key])
        #print(args)
        #sys.exit() 
    for z_prec in z_precs:
        subs = []
        setattr(args, 'z_prec', float(z_prec))
        for batch_counter in range(int(len(z_means) / batch)):
            z_means_s = z_means[batch_counter * batch:(batch_counter+1) * batch]
            for z_mean in z_means_s:
                setattr(args, 'z_mean', float(z_mean))
                test_name_new = os.path.join(path, 'train_{}'.format(folder_counter))
                setattr(args, 'test_name', test_name_new)

                proc = ['python3', 'critique.py']
                for k , v in vars(args).items():
                    proc.append('-{}'.format(k))
                    proc.append('{}'.format(v))

                sub = subprocess.Popen(proc)
                subs.append(sub)
                folder_counter += 1
        
            for proc in subs:
                proc.wait()
 

# get best model in nested cv
def best_model(path):
    
    folders = os.listdir(path)
    folders = [f for f in folders if 'train' in f]
    folders = sorted(folders, key=natural_key)

    # get performance for each model in a fold
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
    best_epoch = arg_perf[np.argmax(perf)]
    # best_folder is not necessarily best_run
    return (best_score, best_run, best_epoch, folders[best_run])
    
# TODO: clean this up, it's bad
if __name__ == '__main__':
    args = get_args()
    cv_tune_name = args.cv_tune_name
    folds = args.folds

    # search through all folders
    models_folder = os.path.join('results', cv_tune_name)
    tune_names = os.listdir(models_folder)
    #names = ['pop', 'random', 'sim_1', 'sim_5']
    #names = ['direct_multi_hits']
    names = [args.name]
    for name in names:

        for tune_name in tune_names:
            #print(tune_name)
            for i in range(folds):
            #    print(i)
                #path = os.path.join(models_folder, tune_name, 'fold_{}'.format(i), args.name)
                path_higher = os.path.join(models_folder, tune_name, 'fold_{}'.format(i))
                print(os.listdir(path_higher))
                    
                if name in os.listdir(path_higher):
                    path = os.path.join(models_folder, tune_name, 'fold_{}'.format(i), name)
                    (best_score, best_run, best_epoch, best_folder) = best_model(path)
                    print('best score: {}, best run: {}, best epoch: {}, best folder: {}'.format(best_score, best_run, best_epoch, best_folder))
                    test_fold(path, tune_name, best_folder, best_epoch)
    

  