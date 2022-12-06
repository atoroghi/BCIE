import os, sys, argparse
from proc import dataset_fold
from tune import tuner
from launch import get_model_args, model_arg_asserts
from critique import get_args_critique, crit_arg_asserts

# non-tunable hyperparameters for the training procedure
def get_metamodel_args():
    parser = argparse.ArgumentParser()
    # other hyper-params
    parser.add_argument('-tune_name', default=None, type=str, help="tuner process name")
    parser.add_argument('-model_type', default='simple', type=str, help="model type (svd, Simple, etc)")
    parser.add_argument('-reg_type', default='tilt', type=str, help="tilt or gauss")
    parser.add_argument('-loss_type', default='gauss', type=str, help="softplus or gauss")
    parser.add_argument('-reduce_type', default='sum', type=str, help="sum or mean")
    parser.add_argument('-optim_type', default='adagrad', type=str, help="adagrad or adam")
    parser.add_argument('-sample_type', default='split_reg', type=str, help="single or double (double treats head and tail dists differently)")
    parser.add_argument('-init_type', default='uniform', type=str, help="uniform or normal")
    parser.add_argument('-learning_rel', default='learn', type=str, help="learn or freeze")
    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")
    parser.add_argument('-type_checking', default='check', type=str, help="check or no")
    parser.add_argument('-fold', default=0, type=int, help="fold for running inner cv on")
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    parser.add_argument('-cluster_check', default=False, type=bool, help='run fast version of code')
 
    args = parser.parse_args()
    return args

# non-tunable hyperparameters for critiquing procedure 
def get_metacrit_args():
    parser = argparse.ArgumentParser()
    # other hyper-params
    parser.add_argument('-tune_name', default=None, type=str, help="tuner process name")
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    parser.add_argument('-critique_target', default='single', type=str, help='single or multi')
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gaussian')
    parser.add_argument('-critique_mode', default='random', type=str, help='random or pop or diff')
    parser.add_argument('-map_finder', default='cvx', type= str, help='cvx or gd')
    parser.add_argument('-fold', default=0, type=int, help="fold for running inner cv on")
    parser.add_argument('-num_users', default=1000, type=int, help='number of users')
    parser.add_argument('-cluster_check', default=False, type=bool, help='run fast version of code')

    args = parser.parse_args()
    return args

# update args to fill args with contents of meta args
def update_args(meta_args, args):
    for ma in vars(meta_args):
        for la in vars(args): 
            if ma == la: 
                setattr(args, la, getattr(meta_args, ma))

if __name__ == '__main__':
    # for quick testing
    cluster_check = True # TODO: do cluster check for training embeddings
    cv_tune_name = 'tuned'

    # hp tuning parameters
    n = 10000
    batch = 4
    folds = 5 if not cluster_check else 1
    epochs = 60 // 4 if not cluster_check else 2
    tune_type = 'two_stage' # joint or two_stage

    # get all args
    meta_crit_args = get_metacrit_args()
    meta_model_args = get_metamodel_args()
    crit_args = get_args_critique()
    model_args = get_model_args()

    # set cluser_checks
    meta_crit_args.cluster_check = cluster_check
    meta_model_args.cluster_check = cluster_check

    # outer loop for running with all small and large embedding files (only when tune_type == 'two_stage)
    if tune_type == 'two_stage':
        models_folder = os.path.join('results', cv_tune_name)
        tune_names = os.listdir(models_folder)
    elif meta_crit_args.tune_type == 'joint':
        tune_names = [meta_crit_args.tune_name]

    # run each folder
    for i, tune_name in enumerate(tune_names):
        if cluster_check and i > 0: break
        meta_crit_args.tune_name = tune_name
        meta_model_args.tune_name = tune_name
        
        # do asserts
        model_arg_asserts(model_args)
        crit_arg_asserts(crit_args)

        # TODO: save this elsewehre
        # make gp dir
        #os.makedirs('gp/{}/{}'.format(cv_tune_name, meta_model_args.tune_name), exist_ok=True)

        # update args with non-tunable params
        update_args(meta_crit_args, crit_args)
        update_args(meta_model_args, model_args)

        # package before sending
        meta_args = (meta_crit_args, meta_model_args)
        args = (crit_args, model_args)

        # iterate through each fold
        for fold in range(folds):
            p_tune_name = os.path.join(cv_tune_name, tune_name)
            tuner(meta_args, args, p_tune_name, fold, epochs, batch, n, tune_type) # main tune loop

############
# code to make new dataset split
#    if False: 
        #print('making new datasets')
        #dataset_fold(5)
