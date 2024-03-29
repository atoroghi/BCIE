
from trainer import train
from tester import test
from dataload import DataLoader

import torch, argparse, time, os, sys, yaml, math
import numpy as np

def get_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help="folder for test results")
    parser.add_argument('-tune_name', default='dev_nest', type=str, help="tuner process name")
    parser.add_argument('-upper_tune_name', default='tuned', type=str, help="upper folder that includes tune process files")
    
    # hyper-parameters (optimized)
    parser.add_argument('-lr', default=1, type=float, help="learning rate")
    parser.add_argument('-batch_size', default=4096, type=int, help="batch size")
    parser.add_argument('-emb_dim', default=14, type=int, help="embedding dimension")
    parser.add_argument('-reg_lambda', default=1e-3, type=float, help="kg loss reg term")
    parser.add_argument('-kg_lambda', default=1, type=float, help="l2 regularization parameter")   
    parser.add_argument('-neg_ratio', default=30, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_power', default=0.0, type=float, help="power for neg sampling disribution")
    parser.add_argument('-init_scale', default=1, type=float, help="std for normal, gain for uniform")

    # not used rn.
    parser.add_argument('-hinge_margin', default=1, type=float, help="in case of margin loss, margin")
    
    # for svd (now wrmf)
    parser.add_argument('-rank', default=10, type=int, help="rank for svd")
    parser.add_argument('-n_iter', default=5, type=int, help="number of iterations for approx method")
    parser.add_argument('-alpha', default=1, type=int, help="alpha for WRMF")
    parser.add_argument('-lam', default=100, type=int, help="lam for WRMF")
    
    # other hyper-params
    parser.add_argument('-model_type', default='simple', type=str, help="model type (svd, simple, complex, etc)")
    parser.add_argument('-reg_type', default='tilt', type=str, help="tilt or gauss")
    parser.add_argument('-loss_type', default='gauss', type=str, help="softplus or gauss")
    parser.add_argument('-reduce_type', default='sum', type=str, help="sum or mean")
    parser.add_argument('-optim_type', default='adagrad', type=str, help="adagrad or adam")
    parser.add_argument('-sample_type', default='split_reg', type=str, help="combo, split_reg, split_rev")
    parser.add_argument('-init_type', default='uniform', type=str, help="uniform or normal")
    parser.add_argument('-learning_rel', default='learn', type=str, help="learn or freeze")
    parser.add_argument('-type_checking', default='check', type=str, help="check or no")
    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")
    
    #parser.add_argument('-type_checking',default='no', type=str, help="doing type checking or not")

    # optimization, saving and data
    parser.add_argument('-epochs', default=30, type=int, help="number of epochs")
    parser.add_argument('-save_each', default=1, type=int, help="validate every k epochs")
    parser.add_argument('-dataset', default='AB', type=str, help="dataset name")
    parser.add_argument('-stop_width', default=4, type=int, help="number of SAVES where test is worse for early stopping")
    #parser.add_argument('-fold', default=0, type=int, help="fold to use data from")

    #redundant because of critique
    parser.add_argument('-multi_k', default=10, type=int, help='number of samples for multi type update')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-num_users', default=1000, type=int, help='number of users')
    parser.add_argument('-sim_k', default=0, type=int, help='number closest movies for direct single testing')
    parser.add_argument('-critique_target', default='single', type=str, help='single or multi')
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gauss')
    parser.add_argument('-crit_mode', default='diff', type=str, help='random or pop or diff')
    parser.add_argument('-map_finder', default='cvx', type= str, help='cvx or gd')
    parser.add_argument('-objective', default='hits', type=str, help='hits or rank')

    # redundant args because of inner_cv
    parser.add_argument('-cluster_check', default='False', type=str, help='run fast version of code')
    parser.add_argument('-cv_tune_name', default='tuned', type=str, help='upper level folder name')
    parser.add_argument('-samples', default=10000, type=int, help='no of samples in tuning')
    parser.add_argument('-batch', default=4, type=int, help='no of simultaneous calls of script')
    parser.add_argument('-folds', default=5, type=int, help='no of folds')
    parser.add_argument('-epochs_all', default=120, type=int, help='no of total epochs')
    parser.add_argument('-tune_type', default='two_stage', type=str, help='two_stage or joint')
    parser.add_argument('-param_tuning', default='per_session', type=str, help='per_session or together')
    parser.add_argument('-name', default='diff', type=str, help='name of current test')
    parser.add_argument('-fold', default=0, type=int, help='fold')
    parser.add_argument('-no_hps', default=4, type=int, help='number of considered hps for tuning')
    parser.add_argument('-cv_type', default='crit', type = str, help = 'train or crit')

    args = parser.parse_args()
    return args

def save_hyperparams(path, args):
    dct={}
    for key in args.__dict__:
        val = getattr(args, key)
        dct[key] = val
    
    with open(os.path.join(path, 'info.yml'), 'w') as f:
            yaml.dump(dct, f, sort_keys=False,
                      default_flow_style=False)

def model_arg_asserts(args) :
    assert args.sample_type in ['combo', 'split_reg', 'split_rev']
    assert args.reg_type in ['gauss', 'tilt']
    assert args.loss_type in ['softplus', 'gauss', 'hinge', 'PSL']
    assert args.optim_type in ['adagrad', 'adam']
    assert args.init_type in ['uniform', 'normal']

def main(args):
    model_arg_asserts(args)

    if args.init_scale == 1: 
        print('manual init scale')
        args.init_scale = 6.0 / math.sqrt(args.emb_dim)

    if args.save_each is None:
        args.save_each = args.epochs

    # ensure test_name exits, make test result folder
    if args.test_name == None:
        raise ValueError('enter a test name folder using -test_name')
    os.makedirs('results', exist_ok=True)
    #save_path = os.path.join("results", str(args.test_name))
    save_path = str(args.test_name)
    os.makedirs(save_path, exist_ok=True) 
    save_hyperparams(save_path, args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # print important hyperparameters
    print('lr {:.7f}, init scale: {:.7f}, batch size: {}'. format(args.lr, args.init_scale, args.batch_size))
    dataloader = DataLoader(args)
    
    # this trains and tests
    score = train(dataloader, args, device)

if __name__ == '__main__':
    args = get_model_args()
    main(args)