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
import seaborn as sns

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cv_tune_name', default='tuned', type = str)
    parser.add_argument('-opt', default='test', type = str, help = 'test or hp')
    parser.add_argument('-folds', default=5, type=int, help='no of folds')
    parser.add_argument('-fold', default=0, type=int, help='fold')
    parser.add_argument('-cv_type', default='crit', type = str, help = 'train or crit')
    parser.add_argument('-name', default='diff', type = str, help = 'name of the test')
    parser.add_argument('-type_checking', default='no')
    parser.add_argument('-learnin_rel', default='learn')
    parser.add_argument('-param_tuning', default='together', type=str, help='per_session or together')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    return parser.parse_args() 

def test_fold(path, tune_name, best_folder, best_epoch, cv_type):
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    if cv_type == 'train':
        path = os.path.join(path, best_folder)
        #path = os.path.join('results', tune_name, 'train', 'fold_{}'.format(i), 'train_{}'.format(best_run))
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
                if key in ['session_length', 'multi_k', 'num_users', 'sim_k', 'batch', 'samples', 'session']:
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
def plotter(cv_tune_name, folds):
    models_folder = os.path.join('results', cv_tune_name)
    tune_names = os.listdir(models_folder)
    test_names = [ 'indirect_hits','direct_multi_hits']

    sns.set_theme()
    fig = plt.figure(figsize=(12,8), dpi = 300)
    ax = fig.add_subplot(111)
    for tune_name in tune_names:
        for test_name in test_names:
            hits_10 = np.empty((0,6))
            for fold in range(folds):
                result_path = os.path.join(models_folder, tune_name, 'fold_{}'.format(fold), test_name, 'test_results')
                ranks = np.load(os.path.join(result_path, 'rank_track.npy'))
                hit_10 = np.sum(ranks<12 , axis =0) / (ranks[:,5].shape[0])
                hits_10 = np.append(hits_10, hit_10.reshape(1, hit_10.shape[0]), axis=0)
            hits_10_means = np.mean(hits_10, axis=0)
            hits_10_yerr = 1.96 / np.sqrt(hits_10.shape[0]) * np.std(hits_10, axis=0)
            ax.errorbar(np.arange(hits_10_means.shape[0]), hits_10_means, yerr = hits_10_yerr, fmt='-o', label = test_name)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel('Hits@10', fontsize=16)
        ax.legend()
        plt.savefig(os.path.join(models_folder, tune_name, 'AvgHits.png'))
        sys.exit()


    
# TODO: clean this up, it's bad
if __name__ == '__main__':
    args = get_args()
    cv_tune_name = args.cv_tune_name
    folds = args.folds
    fold = args.fold
    opt = args.opt
    cv_type = args.cv_type # train or crit

    # search through all folders
    models_folder = os.path.join('results', cv_tune_name)
    #plotter(cv_tune_name, folds)
    
    tune_names = os.listdir(models_folder)
    #names = ['pop', 'random', 'sim_1', 'sim_5']
    #names = ['direct_multi_hits']
    names = [args.name]
    for name in names:
        for tune_name in tune_names:

            #for i in range(folds):
            i = fold
            print(i)
            if opt == 'test':

                    #path = os.path.join(models_folder, tune_name, 'fold_{}'.format(i), args.name)
                path_higher = os.path.join(models_folder, tune_name, 'fold_{}'.format(i))
                    
                if name in os.listdir(path_higher):

                    path = os.path.join(models_folder, tune_name, 'fold_{}'.format(i), name)
                    if args.param_tuning == 'per_session':
                        path = os.path.join(path, 'session_{}'.format(args.session_length-1))

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
