#from concurrent.futures import process
import os, sys, torch, gpytorch, argparse, math, subprocess, re, pickle, yaml
import numpy as np
import matplotlib.pyplot as plt
#from varname import nameof
from gp import normal2param, train_sample_gp
from tune_utils import Params, ScriptCall
from outer_cv import best_model 

# sorts files aplpha-numerically
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# main entry point from inner_cv
def tuner(meta_args, args, tune_name, fold, epochs, batch, n):
    (meta_crit_args, meta_model_args) = meta_args
    (crit_args, model_args) = args

    # set folder structure
    path = os.path.join('results', tune_name)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'fold_{}'.format(fold)) 
    os.makedirs(path, exist_ok=True)

    # get and save params required for running main scripts
    model_params = Params('train', model_args, meta_model_args)
    crit_params = Params('crit', crit_args, meta_crit_args)
    params = (crit_params, model_params)
    
    model_params.save()
    crit_params.save()
    dim = len(model_params.param_dict) + len(crit_params.param_dict)

    # main script for launching subprocesses
    script_call = ScriptCall(args, params, tune_name, fold, path)

    # load training data (if something failed)
    if os.path.isfile(os.path.join(path, 'x_train.pt')):
        begin = False
        x_train = torch.load(os.path.join(path, 'x_train.pt'))
        y_train = torch.load(os.path.join(path, 'y_train.pt'))
        print(x_train.shape, y_train.shape)
    else: begin = True

    # main loop
    for e in range(epochs):
        print("epoch: {}".format(e))
        # train models and update points
        if begin:
            train_path = os.path.join(path,'train')
            os.makedirs(train_path , exist_ok=True)
            begin = False
            x_out, score = script_call.train(torch.rand(batch, dim))
            y_train = score
            x_train = x_out

        else:
            # run gaussian process
            x_test = torch.rand((n, dim))
            x_sample = train_sample_gp(x_test, x_train, y_train, batch, dim, e) 
            x_sample[:2] = torch.rand(2, dim)

            x_out, score = script_call.train(x_sample)
            x_train = torch.vstack((x_train, x_out))
            y_train = torch.cat((y_train, score))

            # save training examples to file
            torch.save(x_train, os.path.join(path, 'x_train.pt'))
            torch.save(y_train, os.path.join(path, 'y_train.pt'))

if __name__  == '__main__':
    print('not implimented')
    sys.exit()

# NOTE: old code for loading best model
# if critique, set model to load
#if cv_type == 'crit':
#    (best_score, best_run, best_epoch) = best_model(meta_args.tune_name, 'train', fold)
#    args.load_name = os.path.join('results', meta_args.tune_name, 'train', 'fold_{}'.format(fold), 'train_{}'.format(best_run))
