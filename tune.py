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
def tuner(args, tune_name, fold, epochs, batch, n, tune_type, param_tuning, session_length, session, cv_type):
    (crit_args, model_args) = args

    # set folder structure
    path = tune_name
    os.makedirs(path, exist_ok=True)
    if param_tuning == 'per_session' and 'cv_type' == 'crit':
        session_path = os.path.join(path, 'session_{}'.format(session))
        os.makedirs(session_path, exist_ok=True)

    # get and save params required for running main scripts
    model_params = Params('train', model_args, tune_name, session_length)
    crit_params = Params('crit', crit_args, tune_name, session_length)

    params = (crit_params, model_params)

    if cv_type == 'train':
        model_params.save()
        dim = len(model_params.param_dict)

    elif cv_type == 'crit':
        if tune_type == 'joint':
            model_params.save()
            dim = len(model_params.param_dict) + len(crit_params.param_dict[session])
        elif tune_type == 'two_stage':
            dim = len(crit_params.param_dict[session])
        crit_params.save()

    # main script for launching subprocesses

    script_call = ScriptCall(args, params, tune_name, fold, path, tune_type, param_tuning, session_length, session, cv_type)

    # load training data (if something failed)

    # in the per_session case each session has its own x_train and y_train
    
    if cv_type == 'train':
        gp_path = path
    
    elif cv_type == 'crit':
        if param_tuning == 'per_session':
            gp_path = session_path
        elif param_tuning == 'together':
            gp_path = path
    

    if os.path.isfile(os.path.join(gp_path, 'x_train.pt')):
        begin = False
        x_train = torch.load(os.path.join(gp_path, 'x_train.pt'))
        y_train = torch.load(os.path.join(gp_path, 'y_train.pt'))
    else: begin = True

    # main loop
    for e in range(epochs):
        print("epoch: {}".format(e))
        # train models and update points
        if begin:
            #train_path = os.path.join(path, 'train')
            #crit_path = os.path.join(path, 'crit', )
            #os.makedirs(train_path , exist_ok=True)
            #os.makedirs(crit_path , exist_ok=True)
            begin = False
            x_out, score = script_call.train(torch.rand(batch, dim))
            y_train = score
            x_train = x_out

        else:
            # run gaussian process
            x_test = torch.rand((n, dim))
            x_sample = train_sample_gp(x_test, x_train, y_train, batch, dim, e) 
            if len(x_sample.shape)<2 :
                x_sample = torch.unsqueeze(x_sample, axis = 1)
            x_sample[:2] = torch.rand(2, dim)

            x_out, score = script_call.train(x_sample)
            x_train = torch.vstack((x_train, x_out))
            y_train = torch.cat((y_train, score))

            # save training examples to file
            torch.save(x_train, os.path.join(gp_path, 'x_train.pt'))
            torch.save(y_train, os.path.join(gp_path, 'y_train.pt'))

if __name__  == '__main__':
    print('not implimented')
    sys.exit()
