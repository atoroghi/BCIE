import os, sys
from proc import dataset_fold
from tune import tuner

def eval_best(fold_num):
    # get model, test
    print()

if __name__ == '__main__':
    tune_name = 'adam_tilt_gauss_mean'

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
    for i in range(3, folds):
        tuner(i, epochs, batch, n, tune_name) # main tune loop
    eval_best(i) # run tester on best net of fold i 
