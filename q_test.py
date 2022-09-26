import os, re, yaml, argparse, torch, math, sys
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
    parser.add_argument('-test_name', default='dev')
    parser.add_argument('-type_checking', default='no')
    parser.add_argument('-learnin_rel', default='learn')
    return parser.parse_args() 

def test_fold(tune_name, best_run, best_epoch):
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
    if args.model_type == 'simple':
        load_path = os.path.join(path, 'models', 'best_model.pt')
        model = torch.load(load_path).to(device)
        test(model, dataloader, best_epoch, args, 'test')

    elif args.model_type == 'wrmf':
        wrmf(dataloader, args, 'test', device)
    elif args.model_type == 'critiquing':
        load_path = os.path.join(path, 'models', 'best_model.pt')
        model = torch.load(load_path).to(device)
        critiquing(model, args, 'test')

# TODO: clean this up, it's bad
if __name__ == '__main__':
    tune_name = 'gausslargenegnokg'
    folds = 5
    opt = 'test'

    # search through all folders
    for i in range(folds):
        if opt == 'test':
            path = 'results/{}/fold_{}'.format(tune_name, i)
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
            print('best score: {}, best folder: {}, best epoch: {}'.format(best_score, best_run, best_epoch))
            test_fold(tune_name, best_run, best_epoch)

        elif opt == 'hp':
            load_path = os.path.join('gp', tune_name, 'fold_{}'.format(i))
            hp_ = torch.load(os.path.join(load_path, 'x_train.pt')).numpy()
            y_ = torch.load(os.path.join(load_path, 'y_train.pt')).numpy()

            if i == 0:
                hp = hp_
                y = y_
            else:
                hp = np.concatenate((hp, hp_))
                y = np.concatenate((y, y_))
            print(hp.shape, y.shape)

    if opt == 'hp':
        path = 'results/{}/'.format(tune_name)
        plt.style.use('seaborn')
        hp_names = ['lr', 'batch size', 'emb dim', 
              'reg lambda', 'kg lambda', 'init scale',
              'neg ratio', 'neg power']
        forest = RandomForestRegressor()
        forest.fit(hp, y)
        imp = forest.feature_importances_

        # plot
        arg = np.argsort(imp)[::-1]
        imp = np.sort(imp)[::-1]
        print(imp)
        hp_names = [hp_names[a] for a in arg]
        
        for i in range(imp.shape[0]):
            plt.bar(i, imp[i])

        plt.xticks(ticks=np.arange(0, imp.shape[0]), labels=hp_names)
        plt.ylabel('Random Forest Importance')
        plt.title('Hyperparameter Importance')
        #plt.show()
        plt.savefig(os.path.join(path, 'q_test_hp.jpg'))
        plt.close()
