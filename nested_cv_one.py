import os, sys, argparse
from proc import dataset_fold
from tune import tuner
from launch import get_args

def get_meta_args():
    parser = argparse.ArgumentParser()
    # other hyper-params
    parser.add_argument('-tune_name', default='dev_nest', type=str, help="tuner process name")
    parser.add_argument('-model_type', default='simple', type=str, help="model type (svd, Simple, etc)")
    parser.add_argument('-reg_type', default='tilt', type=str, help="tilt or gauss")
    parser.add_argument('-loss_type', default='softplus', type=str, help="softplus or gauss")
    parser.add_argument('-reduce_type', default='mean', type=str, help="sum or mean")
    parser.add_argument('-optim_type', default='adam', type=str, help="adagrad or adam")
    parser.add_argument('-sample_type', default='double', type=str, help="single or double (double treats head and tail dists differently)")
    parser.add_argument('-init_type', default='normal', type=str, help="uniform or normal")
    parser.add_argument('-kg', default='kg', type=str, help="kg or no_kg")
    parser.add_argument('-learning_rel', default='learn', type=str, help="learn or freeze")
    parser.add_argument('-type_checking', default='no', type=str, help="check or no")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    meta_args = get_meta_args()
    launch_args = get_args()

    # update args with non-tunable params
    for ma in vars(meta_args):
        for la in vars(launch_args):
            if ma == la:
                # back out arg from key
                setattr(launch_args, la, getattr(meta_args, ma))

    # hp tuning parameters
    folds = 5
    epochs = 15
    batch = 4
    n = 10000

    # make train val test split
    if False: 
        print('making new datasets')
        dataset_fold(5)

    # iterate through each fold
    print('tuning')
    i = 1
        # TODO: pass in args
    tuner(i, epochs, batch, n, meta_args.tune_name) # main tune loop