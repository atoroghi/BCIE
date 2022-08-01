import os
import pickle
import numpy as np

def ul_maker(fold):
    path = os.path.join('datasets', 'ML_FB', 'fold {}'.format(fold))
    with open(os.path.join(path, 'ul_train.pkl'), 'rb') as f:
        ul_train = pickle.load(f)
    with open(os.path.join(path, 'ul_test.pkl'), 'rb') as f:
        ul_test = pickle.load(f)
    with open(os.path.join(path, 'ul_val.pkl'), 'rb') as f:
        ul_val = pickle.load(f)
    ul_whole ={}
    for user in list(ul_train.keys()):
        ul_whole[user] = ul_train[user]
        if user in ul_test.keys():
            ul_whole[user] = ul_whole[user] + ul_test[user]
        if user in ul_val.keys():
            ul_whole[user] = ul_whole[user] + ul_val[user]
    with open(path +'/ul_whole', 'wb') as f:
        pickle.dump(ul_whole,f)

for fold in range(0,5):
    ul_maker(fold)