import os, sys, argparse
from proc import dataset_fold
from tune import tuner
from launch import get_model_args, model_arg_asserts
from critique import get_args_critique, crit_arg_asserts

# TODO: passing in of name is terrible 
if __name__ == '__main__':
    # for quick testing
    cluster_check = False 
    cv_tune_name = 'tuned'

    # hp tuning parameters
    n = 10000
    batch = 4
    folds = 5 if not cluster_check else 1
    epochs = 120 // batch if not cluster_check else 2
    tune_type = 'two_stage' # joint or two_stage
    name = 'diff'

    # get all args
    #meta_crit_args = get_metacrit_args()
    #meta_model_args = get_metamodel_args()
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

    # run each folder
    for i, tune_name in enumerate(tune_names):
        if cluster_check and i > 0: break
        
        # do asserts
        model_arg_asserts(model_args)
        crit_arg_asserts(crit_args)

        # package before sending
        args = (crit_args, model_args)

        # iterate through each fold
        for fold in range(folds):
            full_tune_name = os.path.join('results', cv_tune_name, tune_name, 'fold_{}'.format(fold), name)
            tuner(args, full_tune_name, fold, epochs, batch, n, tune_type) # main tune loop

############
# code to make new dataset split
#    if False: 
        #print('making new datasets')
        #dataset_fold(5)
