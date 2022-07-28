import os, sys, re, pickle, torch
import numpy as np
from numpy.random import default_rng

# make adj matrix from rec triplets
def adj_matrix(fold):
    # get data
    main_path = 'datasets/ML_FB/fold {}'.format(fold)
    rec = np.load(os.path.join(main_path, 'train.npy'))
    rec = rec[:, [0,2]]
    user_max = np.max(rec[:,0]) + 1
    item_max = np.max(rec[:,1]) + 1
    print(user_max, item_max)
    
    # make torch sparse array
    rec = torch.from_numpy(rec).T
    v = torch.ones(rec.shape[1])

    s = torch.sparse_coo_tensor(rec, v, (user_max, item_max))

    return s

# remove items from test and val that aren't in train
def remove_new(test, val, train):
    testval = (test, val)
    axis = (0, 2)

    out = []
    # iterate through test or val triplets
    for tv in testval:    
        # remove both users and items that haven't been seen
        for a in axis:
            train_items = np.unique(train[:, a])
            tv_items = np.unique(tv[:, a])
            rm_tv = [item for item in tv_items if item not in train_items]
            for rm in rm_tv:
                tv = np.delete(tv, np.where(tv[:, a] == rm), axis=0)

        out.append(tv)
    return (out[0], out[1])

# user likes for dicts
def user_likes(test, val, train):
    tvt = (test, val, train)

    ul = []
    for data in tvt:
        user_likes = {}
        for i in range(data.shape[0]):
            if data[i,0] not in user_likes:
                user_likes.update({data[i,0]: [data[i,2]]})
            else:
                if data[i,2] not in user_likes[data[i,0]]:
                    user_likes[data[i,0]].append(data[i,2])
        ul.append(user_likes)

    return (ul[0], ul[1], ul[2]) 

# only for splitting rec currently
# assume inner loop has 1 fold
def dataset_fold(num_fold, val_ratio=0.005):
    # split dataset according to the split required
    main_path = 'datasets/ML_FB/'
    rec = np.load(os.path.join(main_path, 'rec.npy'))

    rec = np.random.permutation(rec) # shuffle data 
    fold_len = rec.shape[0] // num_fold # get sizes of each fold

    # make and save each of the folds
    rng = default_rng()
    for i in range(num_fold):
        if i < num_fold - 1: test_inds = np.arange(i * fold_len, (i+1) * fold_len)
        else: test_inds = np.arange(i * fold_len, rec.shape[0])
        
        test = rec[test_inds]
        other = np.delete(rec, test_inds, axis=0) # train + valid data

        # get train and valid from random split
        val_len = int(val_ratio * other.shape[0])
        val_inds = rng.choice(other.shape[0], size=val_len, replace=False)
        
        val = other[val_inds]
        train = np.delete(other, val_inds, axis=0)

        # remove users + items from test and val that aren't in train
        (test, val) = remove_new(test, val, train)

        # build user likes maps
        (ul_test, ul_val, ul_train) = user_likes(test, val, train)

        # save data
        print('saving fold: ', i)
        path = os.path.join(main_path, 'fold {}'.format(i))
        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, 'train.npy'), train, allow_pickle=True)
        np.save(os.path.join(path, 'test.npy'), test, allow_pickle=True)
        np.save(os.path.join(path, 'val.npy'), val, allow_pickle=True)

        with open(os.path.join(path, 'ul_train.pkl'), 'wb') as f:
            pickle.dump(ul_train, f) 
        with open(os.path.join(path, 'ul_test.pkl'), 'wb') as f:
            pickle.dump(ul_test, f) 
        with open(os.path.join(path, 'ul_val.pkl'), 'wb') as f:
            pickle.dump(ul_val, f) 

# for getting main rec and kg
if __name__ == '__main__':
    kg_path = 'datasets/www_data/www_data/Movielens/kg/train.dat'
    rec_path = 'datasets/www_data/www_data/Movielens/rs/ratings.txt'
    kg = np.genfromtxt(kg_path, delimiter='\t', dtype=np.int32)
    rec = np.genfromtxt(rec_path, delimiter='\t', dtype=np.int32)

    rec = rec[:,:3] # remove time col.
    rec[:,2] = rec[:,2] >= 4 # binary ratings, 0 if [0, 4), 1 if [4, 5] 
    rec = rec[rec[:,2] == 1] # select only positive ratings
    rec[:,2] = 0 # set redundant col to relationship 0
    kg[:,1] += 1 # offset
    rec = rec[:, [0,2,1]] # <user, likes, item> format

    TOTAL_FB_IDS = np.max(kg) # total number of default kg pairs (# rel << # entities)

    # paths for converting data
    item2kg_path = 'datasets/www_data/www_data/Movielens/rs/i2kg_map.tsv'
    emap_path = 'datasets/www_data/www_data/Movielens/kg/e_map.dat'

    # maps movie lense id's to free base html links
    ml2fb_map = {}
    with open(item2kg_path) as f:
        for line in f:
            ml_id = re.search('(.+?)\t', line)
            fb_http = re.search('\t(.+?)\n', line)
            
            ml2fb_map.update({int(ml_id.group(1)) : fb_http.group(1)})

    # maps free base html links to free base id's (final format)
    id2html_map = {}
    fb2id_map = {}
    with open(emap_path) as f:
        for kg_id, line in enumerate(f):
            fb_http = re.search('\t(.+?)\n', line)
            
            fb2id_map.update({fb_http.group(1) : kg_id})
            id2html_map.update({kg_id : fb_http.group(1)})

    # convert movielens id's to freebase id's
    i = 0
    while True:
        if i == rec.shape[0]:
            break

        if rec[i,2] in ml2fb_map: 
            # get correct freebase id from data
            fb_http = ml2fb_map[rec[i,2]]
            fb_id = fb2id_map[fb_http]
            # TODO: is this right
            rec[i,2] = fb_id
            i += 1
        # remove from rec (only use movies that are in kg)
        else:
            rec = np.delete(rec, i, axis=0)

    umap_path = 'datasets/www_data/www_data/Movielens/rs/u_map.dat'

    # maps movielens user id's to freebase id's
    userid2fbid_map = {}
    new_ids = 0
    with open(umap_path) as f:
        for line in f:

            ml_id = re.search('\t(.+?)\n', line)
            if int(ml_id.group(1)) in rec[:,0]:
                new_ids += 1
                userid2fbid_map.update({int(ml_id.group(1)) : TOTAL_FB_IDS + new_ids})

    # convert movielens user id's into freebase id's
    for i in range(rec.shape[0]):
        rec[i,0] = userid2fbid_map[rec[i,0]]

    NEW_USER_IDS = new_ids

    # save full kg and rec
    # break up into train / test / val later..
    np.save('datasets/ML_FB/rec.npy', rec, allow_pickle=True)
    np.save('datasets/ML_FB/kg.npy', kg, allow_pickle=True)