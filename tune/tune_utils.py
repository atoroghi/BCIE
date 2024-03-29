import os, re, sys, yaml, torch, subprocess, copy
sys.path.append('..')
import numpy as np
from tune.gp import normal2param
from outer_cv import best_model

# sorts files aplpha-numerically
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# converts parameters in [0, 1] to actual ranges
class Params:
    def __init__(self, cv_type, args, tune_name, session_length):
        self.tune_name = tune_name
        self.cv_type = cv_type
        self.no_hps = args.no_hps


        # don't tune hp's that we don't use
        if cv_type == 'crit': 
            temp_dict = {}
            temp_dict_f = {}
            # we need one set of parameters for each session
            temp_dict[0] = {
                    # covar [1e-5, 1]
                    'user_prec' : ([-5, 5], float, 10),
                    'default_prec' : ([-5, 5], float, 10),
                    #'multi_k' : ([1, 100], int, None)
                    #'z_prec' : ([-5, 5], float, 10),
                    #'z_mean' : ([-5, 5], float, 10),
                    #'etta' : ([-5, 5], float, 10),
                    #'etta_1' : ([-5, 5], float, 10),
                    #'etta_2' : ([-5, 5], float, 10),
                    #'etta_3' : ([-5, 5], float, 10),
                    #'etta_4' : ([-5, 5], float, 10),
                    #'alpha' : ([-5, 0], float, 10),
                }
            part = list(temp_dict[0].keys())[:self.no_hps]
            temp_dict_f[0] = {k: temp_dict[0][k] for k in part}
            for i in range(1,session_length):
                temp_dict[i]= {
                    # covar [1e-5, 1]
                    'default_prec' : ([-5, 5], float, 10),
                    #'user_prec' : ([-5, 5], float, 10),
                    #'multi_k' : ([1, 100], int, None)
                    #'z_prec' : ([-5, 5], float, 10),
                    #'z_mean' : ([-5, 5], float, 10),
                    #'etta' : ([-5, 5], float, 10),
                    #'etta_1' : ([-5, 5], float, 10),
                    #'etta_2' : ([-5, 5], float, 10),
                    #'etta_3' : ([-5, 5], float, 10),
                    #'etta_4' : ([-5, 5], float, 10),
                    #'alpha' : ([-5, 0], float, 10),
                }
                part = list(temp_dict[i].keys())[:self.no_hps -1]
                temp_dict_f[i] = {k: temp_dict[i][k] for k in part}
            self.param_dict = temp_dict_f




            if 'z_prec' in self.param_dict[0]:
                assert args.evidence_type == 'indirect'
            if 'etta' in self.param_dict[0]:
                assert args.update_type == 'laplace'
            if 'multi_k' in self.param_dict[0]:
                assert args.critique_target == 'multi'

        elif cv_type == 'train':
            if args.model_type == 'svd' or args.model_type == 'pop':
                self.param_dict = {
                    # name : (range, type, base)
                    'rank' : ([2, 9], int, 2),
                    'n_iter' : ([1, 200], int,None),
                }
            elif args.model_type == 'wrmf':
                self.param_dict = {
                    #name : (range, type, base)
                    'rank' : ([2, 8], int, 2),
                    'lam' : ([2, 8], int, 2),
                }

            elif args.model_type == 'simple' or args.model_type == 'complex':
                self.param_dict = {
                    # name : (range, type, base)
                    'lr' : ([-3, 1], float, 10),
                    #'init_scale' : ([-6, 1], float, 10),
                    'batch_size' : ([6, 9], int, 2),
                    #'neg_power' : ([0, 1], float, None),
                    'emb_dim' : ([4, 7], int, 2),
                    #'reg_lambda' : ([-7, 1], float, 10),
                    #'kg_lambda' : ([-7, 1], float, 10),
                    'neg_ratio' : ([10, 20], int, None),
                }

    # take params from gp to real values
    def convert(self, i, po, p, args, session, offset=0):
        if self.cv_type == 'train':
            param_dict_type = self.param_dict
        else:
            param_dict_type = self.param_dict[session]
        for j, (arg_name, spec) in enumerate(param_dict_type.items()):
            for a in vars(args):
                # get proper arg corresponding to param_dict values
                if a == arg_name:
                    # back out arg from key
                    out, po[i,j+offset] = normal2param(p[i,j], spec)
                    setattr(args, a, out)
                    
        return args, po

    # save to yaml file 
    def save(self):
        # convert dict specs to strings
        save_dict = {}
        for k, v in self.param_dict.items():
            save_dict.update({k : str(v)})

        with open(os.path.join(self.tune_name, '{}_hp_ranges.yml'.format(self.cv_type)), 'w') as f:
                yaml.dump(save_dict, f, sort_keys=False,
                        default_flow_style=False)

# main class to launch code that gp is trying to opimize
class ScriptCall:
    def __init__(self, args, params, tune_name, fold_num, path, tune_type, param_tuning, session_length, session, cv_type):
        self.args = args
        self.tune_name_temp = tune_name
        self.fold_num = fold_num
        self.params = params
        self.path = os.path.join(path, '..')
        self.tune_type = tune_type
        self.param_tuning = param_tuning
        self.session_length = session_length
        self.session = session
        self.cv_type = cv_type

    def run_process(self,p, py_file, args_list, crit_test_names, crit_load_names, model_test_names):
        subs = []
        for i in range(p.shape[0]):
            args = args_list[i]
            args[0].test_name = crit_test_names[i]
            args[0].load_name = crit_load_names[i]
            

            if py_file == 'launch.py': 
                args[1].test_name = model_test_names[i]
                used_args = args[1]
            elif py_file == 'critique.py': used_args = args[0]

            proc = ['python3', py_file]
            for k, v in vars(used_args).items():
                proc.append('-{}'.format(k))
                proc.append('{}'.format(v))

            sub = subprocess.Popen(proc)
            subs.append(sub)

        for proc in subs:
            proc.wait() 

    # run training process
    def train(self, p):
        p = p.numpy()
        po = np.empty_like(p)
        args_list = []
        crit_test_names, crit_load_names, model_test_names = [], [], []
        train_path = os.path.join(self.path, 'train')

        # params[0] has cv_type == crit and param[1] has cv_type ==1 so we need to do this twice
        for i in range(p.shape[0]): 
            # convert gp [0, 1] to proper parameter vals
            # po is gp values corresponding to discretization
            
            # folder names for loading and saving
            if self.cv_type == 'train':
                folders = sorted(os.listdir(self.tune_name_temp), key=natural_key)
                folders = [f for f in folders if 'train' in f]
                self.tune_name = self.tune_name_temp
                model_test_name = os.path.join(self.tune_name, 'train_{}'.format(len(folders) + i))

            if self.cv_type == 'crit':
                if self.tune_type == 'joint':
                    folders = sorted(os.listdir(os.path.join(self.tune_name_temp, 'train')), key=natural_key)
                    folders = [f for f in folders if 'train' in f]
                    crit_load_name = os.path.join('results', self.tune_name_temp, 'fold_{}'.format(self.fold_num), 'train', 'train_{}'.format(len(folders) + i)) 

                # here we only load the best model of the fold and make crit result folders
                elif self.tune_type == 'two_stage':
                    if self.args[0].param_tuning == 'per_session':
                        self.tune_name = os.path.join(self.tune_name_temp, 'session_{}'.format(self.session))

                    elif self.args[0].param_tuning == 'together':
                        self.tune_name = self.tune_name_temp

                folders = sorted(os.listdir(self.tune_name), key=natural_key)
                folders = [f for f in folders if 'train' in f]

                model_test_name = os.path.join(self.tune_name,'train', 'train_{}'.format(len(folders) + i))
# changing this for now
            (best_score, best_run, best_epoch, best_folder) = best_model(train_path)
            crit_load_name = os.path.join(train_path, best_folder)
            crit_test_name = os.path.join(self.tune_name, 'train_{}'.format(len(folders) + i))
            

            # save to list
            crit_test_names.append(crit_test_name)
            crit_load_names.append(crit_load_name)
            model_test_names.append(model_test_name)

            ## if crit vs if train\
            if self.cv_type == 'crit': 

                # discretize params before feeding back to gp 
                offset = len(self.params[0].param_dict[self.session])

                crit_args, po = self.params[0].convert(i, po, p, self.args[0], self.session)
                model_args = []
                if self.tune_type == 'joint':
                    model_args, po = self.params[1].convert(i, po, p, self.args[1], offset)

                crit_args_copy = copy.deepcopy(crit_args)
                model_args_copy = copy.deepcopy(model_args)

                args_list.append((crit_args_copy, model_args_copy))
            
            elif self.cv_type == 'train':
                offset = len(self.params[1].param_dict)
                model_args = []
                crit_args, po = self.params[0].convert(i, po, p, self.args[0], self.session)
                model_args, po = self.params[1].convert(i, po, p, self.args[1], offset)
                model_args_copy = copy.deepcopy(model_args)
                args_list.append((crit_args, model_args_copy))

  
            # run script for all hyperparams in the batch   
        if self.tune_type == 'joint' and self.cv_type == 'crit':
            print('training')
            self.run_process(p, 'launch.py', args_list, crit_test_names, crit_load_names, model_test_names)
            print('critique')
            self.run_process(p, 'critique.py', args_list, crit_test_names, crit_load_names, model_test_names)
        
        elif self.tune_type == 'two_stage' and self.cv_type == 'crit':
            print('critique')
            self.run_process(p, 'critique.py', args_list, crit_test_names, crit_load_names, model_test_names)
        
        elif self.cv_type == 'train':
            print('training')
            self.run_process(p, 'launch.py', args_list, crit_test_names, crit_load_names, model_test_names)

        ## get recall at k
        best_mrr = torch.empty(p.shape[0])
        for i in range(p.shape[0]):
            load_path = os.path.join(self.tune_name, 'train_{}'.format(len(folders) + i), 'stop_metric.npy')
            mrr = np.load(load_path)
            best_mrr[i] = np.max(mrr)

        po = torch.from_numpy(po)
        print(best_mrr)
        return po, best_mrr 