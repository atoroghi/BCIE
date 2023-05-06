import os, sys, argparse
from proc import dataset_fold
from tune.tune import tuner
from launch import get_model_args, model_arg_asserts
from critique import get_args_critique, crit_arg_asserts

# args
#  n, batch, fold, tune_type etc
# crit_sim_k, crit_mode

# 1) one big set of args
# 2) 

def get_args_inner():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cluster_check', default='False', type=str, help='run fast version of code')
    parser.add_argument('-cv_tune_name', default='tuned', type=str, help='upper level folder name')
    parser.add_argument('-samples', default=10000, type=int, help='no of samples in tuning')
    parser.add_argument('-batch', default=4, type=int, help='no of simultaneous calls of script')
    parser.add_argument('-folds', default=5, type=int, help='no of folds')
    parser.add_argument('-fold', default=0, type=int, help='fold')
    parser.add_argument('-epochs_all', default=2, type=int, help='no of total epochs')
    parser.add_argument('-tune_type', default='two_stage', type=str, help='two_stage or joint')
    parser.add_argument('-param_tuning', default='together', type=str, help='per_session or together')
    parser.add_argument('-name', default='diff', type=str, help='name of current test')
    parser.add_argument('-cv_type', default='crit', type = str, help = 'train or crit')

    # critique args
    parser.add_argument('-multi_k', default=10, type=int, help='number of samples for multi type update')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-num_users', default=1000, type=int, help='number of users')
    parser.add_argument('-sim_k', default=0, type=int, help='number closest movies for direct single testing')
    parser.add_argument('-objective', default='hits', type=str, help='hits or rank or pcd')
    # single vs mult
    parser.add_argument('-critique_target', default='multi', type=str, help='single or multi')
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    parser.add_argument('-no_hps', default=4, type=int, help='number of considered hps for tuning')
    # likelihood
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gauss')
    parser.add_argument('-crit_mode', default='diff', type=str, help='random or pop or diff')
    parser.add_argument('-map_finder', default='cvx', type= str, help='cvx or gd')

    # model args
    parser.add_argument('-model_type', default='simple', type=str, help="model type (svd, Simple, etc)")
    parser.add_argument('-reg_type', default='tilt', type=str, help="tilt or gauss")
    parser.add_argument('-loss_type', default='gauss', type=str, help="softplus or gauss")
    parser.add_argument('-reduce_type', default='sum', type=str, help="sum or mean")
    parser.add_argument('-optim_type', default='adagrad', type=str, help="adagrad or adam")
    parser.add_argument('-sample_type', default='split_reg', type=str, help="combo, split_reg, split_rev")
    parser.add_argument('-init_type', default='uniform', type=str, help="uniform or normal")
    parser.add_argument('-learning_rel', default='learn', type=str, help="learn or freeze")
    parser.add_argument('-type_checking', default='check', type=str, help="check or no")
    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")
    # optimization, saving and data
    parser.add_argument('-epochs', default=30, type=int, help="number of epochs")
    parser.add_argument('-save_each', default=1, type=int, help="validate every k epochs")
    parser.add_argument('-dataset', default='AB', type=str, help="ML_FB or LFM or AB")
    parser.add_argument('-stop_width', default=4, type=int, help="number of SAVES where test is worse for early stopping")
    #parser.add_argument('-fold', default=0, type=int, help="fold to use data from")
    

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # for quick testing
    #cluster_check = False 
    #cv_tune_name = 'tuned'

    # hp tuning parameters
    #n = 10000
    #batch = 4
    #folds = 5 if not cluster_check else 1
    #epochs = 120 // batch if not cluster_check else 2
    #tune_type = 'two_stage' # joint or two_stage
    #name = 'diff'

    inner_args = get_args_inner()
    cluster_check = inner_args.cluster_check
    cv_tune_name = inner_args.cv_tune_name
    n = inner_args.samples
    batch = inner_args.batch
    #folds = inner_args.folds if not cluster_check else 1
    if cluster_check == 'False':
        folds = inner_args.folds
        epochs = inner_args.epochs_all // inner_args.batch
    
    else:
        folds = 1
        epochs = 2
    #folds = inner_args.folds if not cluster_check else 1
    #epochs = inner_args.epochs_all // inner_args.batch if not inner_args.cluster_check else 2
    tune_type = inner_args.tune_type
    name = inner_args.name

    # get all args
    
    crit_args = get_args_critique()
    model_args = get_model_args()

    # set cluser_checks
    crit_args.cluster_check = cluster_check
    model_args.cluster_check = cluster_check

    # outer loop for running with all small and large embedding files (only when tune_type == 'two_stage)
    if tune_type == 'two_stage':
        models_folder = os.path.join('results', cv_tune_name)
        tune_names = os.listdir(models_folder)
    elif tune_type == 'joint':
        tune_names = [crit_args.tune_name]
    tune_names = ['tilt_small']


    # run each folder
    for i, tune_name in enumerate(tune_names):
        #if cluster_check and i > 0: break
        #if cluster_check == 'True':
        #    if i>0: break
        inner_args = get_args_inner()
        crit_args = get_args_critique()
        crit_args.crit_mode = inner_args.crit_mode
        crit_args.sim_k = inner_args.sim_k
        crit_args.param_tuning = inner_args.param_tuning
        cv_type = inner_args.cv_type; param_tuning = inner_args.param_tuning; session_length = inner_args.session_length


        # do asserts
        model_arg_asserts(model_args)
        crit_arg_asserts(crit_args)

        # package before sending
        args = (crit_args, model_args)

        # iterate through each fold
        #for fold in range(folds):
        fold = inner_args.fold
        full_tune_name = os.path.join('results', cv_tune_name, tune_name, 'fold_{}'.format(fold), name)
        # Using the same hps (default_prec, z_prec, etc) in all sessions
        if inner_args.param_tuning == 'together':
            session = 0
            tuner(args, full_tune_name, fold, epochs, batch, n, tune_type, param_tuning, session_length, session, cv_type)
        # Using separate hps for each session
        elif inner_args.param_tuning == 'per_session':
            for session in range(session_length):
                args[0].session = session
                tuner(args, full_tune_name, fold, epochs, batch, n, tune_type, param_tuning, session_length, session, cv_type)



 

####    ########
# code to make new dataset split
#    if False: 
        #print('making new datasets')
        #dataset_fold(5)
