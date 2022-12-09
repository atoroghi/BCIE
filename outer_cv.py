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
    parser.add_argument('-cv_tune_name', default='tuned', type = str)
    parser.add_argument('-opt', default='test', type = str, help = 'test or hp')
    parser.add_argument('-folds', default=5, type=int, help='no of folds')
    parser.add_argument('-cv_type', default='crit', type = str, help = 'train or crit')
    parser.add_argument('-name', default='diff', type = str, help = 'name of the test')
    parser.add_argument('-type_checking', default='no')
    parser.add_argument('-learnin_rel', default='learn')
    return parser.parse_args() 

def test_fold(path, tune_name, best_folder, best_epoch, cv_type):
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    if cv_type == 'train':
        path = os.path.join('results', tune_name, 'train', 'fold_{}'.format(i), 'train_{}'.format(best_run))
        with open(os.path.join(path, 'info.yml'), 'r') as f:
            yml = yaml.safe_load(f)
            for key in yml.keys():
                setattr(args, key, yml[key])
        dataloader = DataLoader(args)
        if args.model_type == 'simple':
            load_path = os.path.join(path, 'models', 'best_model.pt')
            model = torch.load(load_path).to(device)
            test(model, dataloader, best_epoch, args, 'test')

        elif args.model_type == 'wrmf':
            wrmf(dataloader, args, 'test', device)
    if cv_type == 'crit':
        save_path = os.path.join(path, 'test_results')
        os.makedirs(save_path, exist_ok = True)
        path = os.path.join(path, best_folder)

        with open(os.path.join(path, 'crit hps.yml'), 'r') as f:
            yml = yaml.safe_load(f)
            for key in yml.keys():
                #TODO: fix this, while saving the yml file the numerical values shouldn't be strings
                if key in ['session_length', 'multi_k', 'num_users', 'sim_k', 'batch', 'samples']:
                    setattr(args, key, int(yml[key]))
                else:
                    try:
                        setattr(args, key, float(yml[key]))
                    except:
                        setattr(args, key, yml[key])

        setattr(args, 'test_name', save_path)
        print("results are being saved in:", args.test_name)
        critiquing(args, 'test')

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
    opt = args.opt
    cv_type = args.cv_type # train or crit

    # search through all folders
    models_folder = os.path.join('results', cv_tune_name)
    tune_names = os.listdir(models_folder)
    for tune_name in tune_names:
        for i in range(folds):
            print(i)
            if opt == 'test':
                path = os.path.join(models_folder, tune_name, 'fold_{}'.format(i), args.name)
                if args.name in os.listdir(path):
                    (best_score, best_run, best_epoch, best_folder) = best_model(path)
                    print('best score: {}, best run: {}, best epoch: {}, best folder: {}'.format(best_score, best_run, best_epoch, best_folder))
                    test_fold(path, tune_name, best_folder, best_epoch, cv_type)
    

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
    sys.exit()

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
        plt.savefig(os.path.join(path, 'q_test_hp.jpg'))
        plt.close()
