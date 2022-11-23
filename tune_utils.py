import os, re, sys, yaml, torch, subprocess
import numpy as np
from gp import normal2param

# sorts files aplpha-numerically
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# converts parameters in [0, 1] to actual ranges
class Params:
    def __init__(self, cv_type, args, meta_args):
        self.tune_name = meta_args.tune_name
        self.cv_type = cv_type

        # TODO: don't tune hp's that we don't use
        if cv_type == 'crit': 
            self.param_dict = {
                # covar [1e-5, 1]
                'default_prec' : ([-5, 5], float, 10),
                'user_prec' : ([-5, 5], float, 10),
                #'multi_k' : ([1, 100], int, None)
                #'z_prec' : ([-5, 5], float, 10),
                #'etta_0' : ([-5, 5], float, 10),
                #'etta_1' : ([-5, 5], float, 10),
                #'etta_2' : ([-5, 5], float, 10),
                #'etta_3' : ([-5, 5], float, 10),
                #'etta_4' : ([-5, 5], float, 10),
                #'alpha' : ([-5, 0], float, 10),
                #'-tau_z_f': ([-5, 5], float, 10),
                #'-tau_z_inv': ([-5, 5], float, 10),
            }

            # TODO: better asserts to automatically set hps
            if 'z_prec' in self.param_dict:
                assert meta_args.evidence_type == 'indirect'
            if 'etta_0' in self.param_dict:
                assert meta_args.update_type == 'laplace'
            if 'multi_k' in self.param_dict:
                assert meta_args.critique_target == 'multi'

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
                    'init_scale' : ([-6, 1], float, 10),
                    'batch_size' : ([11, 14], int, 2),
                    #'neg_power' : ([0, 1], float, None),
                    #'emb_dim' : ([2, 8], int, 2),
                    #'reg_lambda' : ([-7, 1], float, 10),
                    #'kg_lambda' : ([-7, 1], float, 10),
                    #'neg_ratio' : ([15, 30], int, None),
                }

    # take params from gp to real values
    def convert(self, i, po, p, args):

        for j, (arg_name, spec) in enumerate(self.param_dict.items()):

            # get proper arg corresponding to param_dict values

            for a in vars(args):
                if a == arg_name:
            #            # back out arg from key
                    out, po[i,j] = normal2param(p[i,j], spec)
                    setattr(args, a, out)

        return args, po

    # save to yaml file 
    def save(self):
        # convert dict specs to strings
        save_dict = {}
        for k, v in self.param_dict.items():
            save_dict.update({k : str(v)})

        save_path = os.path.join('results', self.tune_name)
        with open(os.path.join(save_path, '{}_hp_ranges.yml'.format(self.cv_type)), 'w') as f:
                yaml.dump(save_dict, f, sort_keys=False,
                        default_flow_style=False)

# main class to launch code that gp is trying to opimize
class ScriptCall:
    def __init__(self, args, params, tune_name, fold_num, path):
        # TODO: unpack params into model and crit
        self.args = args
        self.tune_name = tune_name
        self.fold_num = fold_num
        self.params = params
        self.path = path


    def run_process(self,p, py_file, args):
        subs = []

        for i in range(p.shape[0]):

            proc = ['python3', py_file]
            for k, v in vars(args).items():
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


        for i in range(p.shape[0]): 
        ##    # convert gp [0, 1] to proper parameter vals
        #    # po is gp values corresponding to discretization
        #    # params[0] has cv_type == crit and param[1] has cv_type ==1 so we need to do this twice
            folders = sorted(os.listdir(os.path.join(self.path, 'train')), key=natural_key)

            folders = [f for f in folders if 'train' in f]
        #    
        #    # self.args[0] is crit_args and self.args[1] is model_args
            self.args[0].test_name = os.path.join(self.tune_name, 'fold_{}'.format(self.fold_num), 'crit', 'train_{}'.format(len(folders) + i)) 
            self.args[0].load_name = os.path.join('results', self.tune_name, 'fold_{}'.format(self.fold_num), 'train', 'train_{}'.format(len(folders) + i)) 
            self.args[1].test_name = os.path.join(self.tune_name, 'fold_{}'.format(self.fold_num), 'train', 'train_{}'.format(len(folders) + i))
#
            self.crit_args, po = self.params[0].convert(i, po, p, self.args[0])
            self.model_args, po = self.params[1].convert(i, po, p, self.args[1])
#

        ##for proc in subs_crit:
        ##    proc.wait()
        self.run_process(p, 'launch.py', self.model_args)
        self.run_process(p, 'critique.py', self.crit_args)
        #print('stopping')
        #sys.exit()
        ##
        ##sys.exit()
#
        ## get recall at k
        best_hits = torch.empty(p.shape[0])
        for i in range(p.shape[0]):
            print("len_folders",len(folders))
            #load_path = os.path.join(path, 'train_{}'.format(len(folders) + i), 'stop_metric.npy')
            #load_path = os.path.join(self.path, 'crit', 'fold_{}'.format(self.args[0].fold), 'train_{}'.format(len(folders) + i), 'stop_metric.npy')
            load_path = os.path.join(self.path, 'crit', 'train_{}'.format(len(folders) + i), 'stop_metric.npy')
            hits = np.load(load_path)
            best_hits[i] = np.max(hits)

        return torch.from_numpy(po), best_hits 