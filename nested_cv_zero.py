import os, sys, argparse
from proc import dataset_fold
from tune import tuner
from launch import get_model_args
from critique import get_args_critique

# TODO: these shouldn't have to be defined in the same file...
# non-tunable hyperparameters for the training procedure
def get_metatrain_args():
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
    parser.add_argument('-evidence_type', default='indirect', type=str, help='direct or indirect')
    parser.add_argument('-critique_target', default='object', type=str, help='object or item')
    parser.add_argument('-tune_name', default='gausstypereg', type=str, help="tuner process name")
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gauss')
    parser.add_argument('-critique_mode', default='random', type=str, help='random or pop or diff')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # TODO: make these into argparse
    # hp tuning parameters
    cv_type = 'crit' # train or crit
    folds = 5
    epochs = 15
    batch = 4
    n = 10000

    assert cv_type in ['crit', 'train']

    # TODO: rename these to be better
    if cv_type == 'crit':
        meta_args = get_metacrit_args()
        args = get_args_critique()
    elif cv_type == 'train':
        meta_args = get_metatrain_args()
        args = get_model_args()

    # make gp dir
    os.makedirs('gp/{}'.format(meta_args.tune_name), exist_ok=True)

    # update args with non-tunable params
    for ma in vars(meta_args):
        for la in vars(args):
            if ma == la:
                setattr(args, la, getattr(meta_args, ma))

    # make train val test split
    if False: 
        print('making new datasets')
        dataset_fold(5)

    # iterate through each fold
    fold = 0
    tuner(cv_type, meta_args, args, fold, epochs, batch, n) # main tune loop