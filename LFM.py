import os, sys, re, pickle, torch
import numpy as np
from numpy.random import default_rng

import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import time
import sys, os
import pickle


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




def make_critiquing_dicts(rec,kg):
    main_path = 'datasets/www_data/www_data/AmazonBook/rs/'
    items = np.unique(rec[:,2])
    popularities={}
    pop_data=np.delete(kg,1,1)
    unique, counts = np.unique(pop_data, return_counts=True)
    # a dictionary containing the popularity of each object
    pop_counts=dict(zip(unique, counts))
    with open(os.path.join(main_path,'pop_counts.pkl'), 'wb') as f:
        pickle.dump(pop_counts, f)
    
    #making dictionary of facts about each item

    items_facts_head={}
    items_facts_tail={}
    for item in items:
      items_facts_head[item]=kg[np.where(kg[:, 0] == item)][:,1:]
      items_facts_tail[item]=kg[np.where(kg[:, 2] == item)][:,:-1]    
    with open(os.path.join(main_path,'items_facts_head.pkl'), 'wb') as f:
        pickle.dump(items_facts_head, f)
    with open(os.path.join(main_path,'items_facts_tail.pkl'), 'wb') as f:
        pickle.dump(items_facts_tail, f)


       #mappings from objects to items
    
    obj2items={}
    for obj in pop_counts.keys():
      if obj not in items:

        objkg = kg[np.where((kg[:, 0] == obj) | (kg[:, 2] == obj))]
        objkg = np.delete(objkg,1,1)
        mapped_items = np.intersect1d(items,objkg)
        obj2items[obj] = mapped_items
      else:
        obj2items[obj] = np.array([obj])
    with open(os.path.join(main_path,'obj2items.pkl'), 'wb') as f:
        pickle.dump(obj2items, f)


# make dictionaries "valid_heads" and "valid_tails" for realations to be used in type checking 
def make_types_dicts(rec,kg):
    data = np.concatenate([rec,kg], axis = 0)
    main_path = 'datasets/www_data/www_data/AmazonBook/rs/'
    valid_heads = {}
    valid_tails = {}
    valid_heads_freq = {}
    valid_tails_freq = {}
    all_rels = np.unique(data[:,1])
    for rel in all_rels:
        heads_all = data[np.where(data[:,1]==rel)[0],0]
        tails_all = data[np.where(data[:,1]==rel)[0],2]
        heads, heads_counts = np.unique(heads_all, return_counts = True)
        tails, tails_counts = np.unique(tails_all, return_counts = True)
        valid_heads[rel] = heads
        valid_tails[rel] = tails
        valid_heads_freq[rel] = heads_counts / np.sum(heads_counts)
        valid_tails_freq[rel] = tails_counts / np.sum(tails_counts)

    with open(os.path.join(main_path, 'valid_heads.pkl'), 'wb') as f:
        pickle.dump(valid_heads, f) 
    with open(os.path.join(main_path, 'valid_tails.pkl'), 'wb') as f:
        pickle.dump(valid_tails, f) 
    with open(os.path.join(main_path, 'valid_heads_freq.pkl'), 'wb') as f:
        pickle.dump(valid_heads_freq, f)
    with open(os.path.join(main_path, 'valid_tails_freq.pkl'), 'wb') as f:
        pickle.dump(valid_tails_freq, f)


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


def dataset_fold(rec, num_fold, val_ratio=0.005):
    # split dataset according to the split required
    main_path = 'datasets/www_data/www_data/AmazonBook/rs/'
    #rec = np.load(os.path.join(main_path, 'rec.npy'))

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






if __name__ == '__main__':
    kg_path = 'datasets/www_data/www_data/AmazonBook/kg/train.dat'
    rec_path = 'datasets/www_data/www_data/AmazonBook/rs/ratings.txt'

    #%

    kg = np.genfromtxt(kg_path, delimiter='\t', dtype=np.int32)
    rec = np.genfromtxt(rec_path, delimiter='\t', dtype=np.int32)
    rec = rec[:,:3] # remove time col.
    rec[:,2] = rec[:,2] >= 4 # binary ratings, 0 if [0, 4), 1 if [4, 5] 
    rec = rec[rec[:,2] == 1] # select only positive ratings
    rec[:,2] = 0 # set redundant col to relationship 0
    kg[:,1] += 1 # offset
    #kg = remove_rare(kg) #remove rare relations
    #kg_train, kg_test = split_kg(kg)
    rec = rec[:, [0,2,1]] # <user, likes, item> format




    TOTAL_FB_IDS = np.max(kg) # total number of default kg pairs (# rel << # entities)
    # paths for converting data
    item2kg_path = 'datasets/www_data/www_data/AmazonBook/rs/i2kg_map.tsv'
    emap_path = 'datasets/www_data/www_data/AmazonBook/kg/e_map.dat'
    # maps movie lense id's to free base html links
    ml2fb_map = {}
    with open(item2kg_path) as f:
        for line in f:
            ml_id = re.search('(.+?)\t', line)
            fb_http = re.search('\t(.+?)\n', line)
            #ml2fb_map.update({int(ml_id.group(1)) : fb_http.group(1)})
            ml2fb_map.update({ml_id.group(1) : fb_http.group(1)})
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
    j = 0
    while True:
        if i == rec.shape[0]:
            break
        if rec[i,2] in ml2fb_map: 
            # get correct freebase id from data
            fb_http = ml2fb_map[rec[i,2]]
            fb_id = fb2id_map[fb_http]
            rec[i,2] = fb_id
            i += 1
        # remove from rec (only use movies that are in kg)
        else:
            rec = np.delete(rec, i, axis=0)
        j += 1
        print("1",j)


    i = 0
    j = 0
    while True:
        if i == rec.shape[0]:
            break
        if rec[i,2] not in kg:
            rec = np.delete(rec, i, axis=0)
        i += 1
        j += 1
        print("2",j)

    np.save('datasets/www_data/www_data/AmazonBook/rs/rec_raw.npy', rec, allow_pickle=True)

    umap_path = 'datasets/www_data/www_data/AmazonBook/rs/u_map.dat'
    userid2fbid_map = {}
    new_ids = 0
    with open(umap_path) as f:
        for line in f:
            ml_id = re.search('\t(.+?)\n', line)
            #if int(ml_id.group(1)) in rec[:,0]:
            if ml_id.group(1) in rec[:,0]:
                new_ids += 1
                userid2fbid_map.update({int(ml_id.group(1)) : TOTAL_FB_IDS + new_ids})
    # convert movielens user id's into freebase id's
    for i in range(rec.shape[0]):
        rec[i,0] = userid2fbid_map[rec[i,0]]
    NEW_USER_IDS = new_ids


    dataset_fold(rec, 5, val_ratio=0.005)

    make_types_dicts(rec,kg)

    make_critiquing_dicts(rec,kg)

    np.save('datasets/www_data/www_data/AmazonBook/rs/rec.npy', rec, allow_pickle=True)
    np.save('datasets/www_data/www_data/AmazonBook/rs/kg.npy', kg, allow_pickle=True)
