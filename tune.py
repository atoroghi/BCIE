from concurrent.futures import process
import os, sys, torch, gpytorch, argparse, math, subprocess, re, pickle, yaml
import numpy as np
import matplotlib.pyplot as plt
from varname import nameof
from gp import normal2param, train_sample_gp
from outer_cv import best_model 

# converts parameters in [0, 1] to actual ranges
class Params:
    def __init__(self, cv_type, args, meta_args):
        self.tune_name = meta_args.tune_name

        if cv_type == 'crit': 
            self.param_dict = {
                'l_prec' : ([-5, 5], float, 10),
                'tau_prior_f' : ([-5, 5], float, 10),
                'tau_prior_inv' : ([-5, 5], float, 10),
                'ettaone' : ([-5, 5], float, 10),
                'ettatwo' : ([-5, 5], float, 10),
                'ettathree' : ([-5, 5], float, 10),
                'ettafour' : ([-5, 5], float, 10),
                #'-tau_z_f': ([-5, 5], float, 10),
                #'-tau_z_inv': ([-5, 5], float, 10),
            }

        elif cv_type == 'train':
            if args.model_type == 'svd':
                self.param_dict = {
                    # name : (range, type, base)
                    'rank' : ([2, 9], int, 2),
                    'n_iter' : ([1, 200], int,None),
                }

            elif args.model_type == 'simple':
                self.param_dict = {
                    # name : (range, type, base)
                    'lr' : ([-6, 1], float, 10),
                    'batch_size' : ([11, 14], int,2),
                    'emb_dim' : ([2, 8], int, 2),
                    #'reg_lambda' : ([-7, 1], float, 10),
                    #'kg_lambda' : ([-7, 1], float, 10),
                    'init_scale' : ([-6, 1], float, 10),
                    'neg_ratio' : ([15, 30], int, None),
                    #'neg_power' : ([0, 1], float, None),
                }

        self.save()

    # take params from gp to real values
    def convert(self, i, po, p, args):
        for j, (arg_name, spec) in enumerate(self.param_dict.items()):
            # get proper arg corresponding to param_dict values
            for a in vars(args):
                if a == arg_name:
                    # back out arg from key
                    out, po[i,j] = normal2param(p[i,j], spec)
                    setattr(args, a, out)

        return args, po

    # save to yaml file 
    def save(self):
        # convert dict specs to strings
        save_dict = {}
        for k, v in self.param_dict.items():
            save_dict.update({k : str(v)})

        save_path = os.path.join('gp', self.tune_name)
        with open(os.path.join(save_path, 'hp_ranges.yml'), 'w') as f:
                yaml.dump(save_dict, f, sort_keys=False,
                        default_flow_style=False)

# sorts files aplpha-numerically
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# main class to launch code that gp is trying to opimize
class Launch:
    def __init__(self, cv_type, args, param, tune_name, fold_num):
        self.args = args
        self.tune_name = tune_name
        self.fold_num = fold_num
        self.param = param
        self.cv_type = cv_type

    # run training process
    def train(self, p):
        p = p.numpy()
        po = np.empty_like(p)

        # make fold folder
        path = os.path.join('results', self.tune_name, self.cv_type, 'fold_{}'.format(self.fold_num))
        os.makedirs(path, exist_ok=True)

        subs = []
        for i in range(p.shape[0]): 
            # convert gp [0, 1] to proper parameter vals
            # po is gp values corresponding to discretization
            self.args, po = self.param.convert(i, po, p, self.args)

            folders = sorted(os.listdir(path), key=natural_key)
            folders = [f for f in folders if 'train' in f]

            # make string to pass arguments
            if self.cv_type == 'crit':
                proc_input = ['python3', 'critique.py']
                self.args.test_name = os.path.join(self.tune_name, 'crit', 'fold_{}'.format(self.fold_num), 'train_{}'.format(len(folders) + i)) 
            else:
                proc_input = ['python3', 'launch.py']
                self.args.test_name = os.path.join(self.tune_name, 'train', 'fold_{}'.format(self.fold_num), 'train_{}'.format(len(folders) + i)) 
            for k, v in vars(self.args).items():
                proc_input.append('-{}'.format(k))
                proc_input.append('{}'.format(v))

            sub = subprocess.Popen(proc_input)
            subs.append(sub)
        
        for proc in subs:
            proc.wait()
        print('done')
        sys.exit()

        # get recall at k
        best_hits = torch.empty(p.shape[0])
        for i in range(p.shape[0]):
            load_path = os.path.join(path, 'train_{}'.format(len(folders) + i), 'stop_metric.npy')
            hits = np.load(load_path)
            best_hits[i] = np.max(hits)

        return torch.from_numpy(po), best_hits 

# main entry point from inner_cv
def tuner(cv_type, meta_args, args, fold, epochs, batch, n):
    # if critique, set model to load
    args.fold = fold
    if cv_type == 'crit':
        (best_score, best_run, best_epoch) = best_model(meta_args.tune_name, fold)
        args.load_name = os.path.join('results', meta_args.tune_name, 'train', 'fold_{}'.format(fold), 'train_{}'.format(best_run))
    
    # build important classes
    param = Params(cv_type, args, meta_args)
    launch = Launch(cv_type, args, param, meta_args.tune_name, fold)
    dim = len(param.param_dict)
    print('tuning with {} parameters'.format(dim))

    # load training data
    path = os.path.join('gp', meta_args.tune_name)
    os.makedirs(path, exist_ok=True)
    gp_path = os.path.join(path, 'fold_{}'.format(fold)) 
    os.makedirs(gp_path, exist_ok=True)

    if os.path.isfile(os.path.join(gp_path, 'x_train.pt')):
        begin = False
        x_train = torch.load(os.path.join(gp_path, 'x_train.pt'))
        y_train = torch.load(os.path.join(gp_path, 'y_train.pt'))
        print(x_train.shape)
        print(y_train.shape)
    else:
        begin = True

    # main loop
    for e in range(epochs):
        # train models and update points
        if begin:
            begin = False
            x_out, score = launch.train(torch.rand(batch, dim))
            y_train = score
            x_train = x_out
        else:
            # run gaussian process
            x_test = torch.rand((n, dim))
            x_sample = train_sample_gp(x_test, x_train, y_train, batch, dim, e) 
            x_sample[:2] = torch.rand(2, dim)

            x_out, score = launch.train(x_sample)
            x_train = torch.vstack((x_train, x_out))
            y_train = torch.cat((y_train, score))

            # save training examples to file
            torch.save(x_train, os.path.join(gp_path, 'x_train.pt'))
            torch.save(y_train, os.path.join(gp_path, 'y_train.pt'))

if __name__  == '__main__':
    print('not implimented')
    sys.exit()

# independent dims
#if __name__ == '__main__':
    #epochs = 100
    #sub_epochs = 1
    #dim = 6
    #n = 1000
    #batch = 4

    #args = get_args()
    #tune = Tune_Param(args)
    
    ## for tracking hp's generated from process
    #saved_hp = {}
    #if os.path.isfile('gp/hp.pkl'):
        #lens = []
        #with open('gp/hp.pkl', 'rb') as f:
            #saved_hp = pickle.load(f)

        #x_all = torch.empty((batch, dim))
        #for e, (x, y) in saved_hp.items():
            #lens.append(x.shape[0])
            #x_all[:,e] = x[np.argmax(y)]
        #start = np.argmin(lens)
    #else:
        #x_all = torch.zeros((batch, dim)) + 0.5
        #start = 0

    ## TODO: when epoch < dim include implicit hp = 0.5 for previous runs... (if better)
    ## main loop
    #for epoch in range(start, epochs):
        #e = epoch % dim # cycle every _ epochs
        #print('epoch: {} e: {}'.format(epoch, e))

        ## collect old params to re-use in gp
        #if e in saved_hp:
            #(x_train, y_train) = saved_hp[e]     
            #print('training data')
            #print(x_train)
            #print(y_train)
            #print()

        ## train models and update points
        #for sub in range(sub_epochs):
            #if e not in saved_hp:
                ## generate random hp and train if no training examples
                #x_all[:,e] = torch.rand(batch)
                #x_out, score = tune.train(x_all)

                #y_train = score
                #x_train = x_out[:,e]
            #else:
                ## run gaussian process
                #x_test, _ = torch.sort(torch.rand(n))
                #x_sample = train_sample_gp(x_test, x_train, y_train, batch, 1, e)               

                ## train nn and get score
                #x_all[:,e] = x_sample
                #x_out, score = tune.train(x_all)

                #x_train = torch.cat((x_train, x_out[:,e]))
                #y_train = torch.cat((y_train, score))

        ## save hp for next major epoch
        #saved_hp.update({e : (x_train, y_train)})
        #with open('gp/hp.pkl', 'wb') as f:
            #pickle.dump(saved_hp, f)

        ## update x_all with best param
        #print('setting new param')
        #best = torch.argmax(y_train)
        #x_all[:,e] = x_train[best]
        #print(x_all)
        #print()
