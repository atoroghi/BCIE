import os, sys, argparse
from proc import dataset_fold
from tune import tuner
from launch import get_model_args, model_arg_asserts
from critique import get_args_critique, crit_arg_asserts

# non-tunable hyperparameters for the training procedure
def get_metamodel_args():
    parser = argparse.ArgumentParser()
    # other hyper-params
    parser.add_argument('-tune_name', default='dev', type=str, help="tuner process name")
    parser.add_argument('-model_type', default='simple', type=str, help="model type (svd, Simple, etc)")
    parser.add_argument('-reg_type', default='gauss', type=str, help="tilt or gauss")
    parser.add_argument('-loss_type', default='gauss', type=str, help="softplus or gauss")
    parser.add_argument('-reduce_type', default='mean', type=str, help="sum or mean")
    parser.add_argument('-optim_type', default='adagrad', type=str, help="adagrad or adam")
    parser.add_argument('-sample_type', default='double', type=str, help="single or double (double treats head and tail dists differently)")
    parser.add_argument('-init_type', default='normal', type=str, help="uniform or normal")
    parser.add_argument('-learning_rel', default='learn', type=str, help="learn or freeze")
    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")
    parser.add_argument('-type_checking', default='no', type=str, help="check or no")
 
    args = parser.parse_args()
    return args

# non-tunable hyperparameters for critiquing procedure 
def get_metacrit_args():
    parser = argparse.ArgumentParser()
    # other hyper-params
    parser.add_argument('-tune_name', default='dev', type=str, help="tuner process name")
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    parser.add_argument('-critique_target', default='single', type=str, help='single or multi')
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gaussian')
    parser.add_argument('-critique_mode', default='random', type=str, help='random or pop or diff')
    parser.add_argument('-map_finder', default='cvx', type= str, help='cvx or gd')

    args = parser.parse_args()
    return args

# update args to fill args with contents of meta args
def update_args(meta_args, args):
    for ma in vars(meta_args):
        for la in vars(args): 
            if ma == la: 
                setattr(args, la, getattr(meta_args, ma))

if __name__ == '__main__':
    # hp tuning parameters
    folds = 5
    epochs = 3
    batch = 4
    n = 10000

    # TODO: rename these to be better
    meta_crit_args = get_metacrit_args()
    meta_model_args = get_metamodel_args()
    crit_args = get_args_critique()
    model_args = get_model_args()

    # run checks
    assert meta_crit_args.tune_name == meta_model_args.tune_name
    tune_name = meta_model_args.tune_name
    model_arg_asserts(model_args)
    crit_arg_asserts(crit_args)

    # make gp dir
    os.makedirs('gp/{}'.format(meta_model_args.tune_name), exist_ok=True)

    # update args with non-tunable params
    update_args(meta_crit_args, crit_args)
    update_args(meta_model_args, model_args)

    # package before sending
    meta_args = (meta_crit_args, meta_model_args)
    args = (crit_args, model_args)

    # iterate through each fold
    for fold in range(folds):
        tuner(meta_args, args, tune_name, fold, epochs, batch, n) # main tune loop

############
# code to make new dataset split
#    if False: 
        #print('making new datasets')
        #dataset_fold(5)
