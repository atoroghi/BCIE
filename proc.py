import os, sys, re, pickle
import numpy as np

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
print(np.unique(rec[:, 2]).shape)

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

# split rec and kg
np.random.shuffle(rec)
split = int(0.7*rec.shape[0])
rec_train = rec[:split]
rec_test = rec[split:]

np.random.shuffle(kg)
kg_train = kg[:split]
kg_test = kg[split:]

# user like maps
user_likes_test = {}
for i in range(rec_test.shape[0]):
    if rec_test[i,0] not in user_likes_test:
        user_likes_test.update({rec_test[i,0]: [rec_test[i,2]]})
    else:
        if rec_test[i,2] not in user_likes_test[rec_test[i,0]]:
            user_likes_test[rec_test[i,0]].append(rec_test[i,2])

user_likes_train = {}
for i in range(rec_train.shape[0]):
    if rec_train[i,0] not in user_likes_train:
        user_likes_train.update({rec_train[i,0]: [rec_train[i,2]]})
    else:
        if rec_train[i,2] not in user_likes_train[rec_train[i,0]]:
            user_likes_train[rec_train[i,0]].append(rec_train[i,2])

# TODO: rename some things here?
# make kg dictionaries
name = ['test', 'train']
heads_test, tails_test = {}, {}
for i, kg in enumerate([kg_test, kg_train]):
    for j, fact in enumerate(kg):
        headkey = tuple(fact[1:])
        tailkey = tuple(fact[:2])
        if headkey in heads_test.keys():
            heads_test[headkey].append(fact[0])
        else:
            heads_test[headkey] = [fact[0]]
        if tailkey in tails_test.keys():
            tails_test[tailkey].append(fact[2])
        else:
            tails_test[tailkey] = [fact[2]]
    with open('datasets/ML_FB/kg_tail_{}.pkl'.format(name[i]), 'wb') as f:
       pickle.dump(heads_test, f)   
    with open('datasets/ML_FB/kg_head_{}.pkl'.format(name[i]), 'wb') as f:
       pickle.dump(tails_test, f)

np.save('datasets/ML_FB/rec_train.npy', rec_train, allow_pickle=True)
np.save('datasets/ML_FB/rec_test.npy', rec_test, allow_pickle=True)
np.save('datasets/ML_FB/kg_train.npy', kg_train, allow_pickle=True)
np.save('datasets/ML_FB/kg_test.npy', kg_test, allow_pickle=True)

with open('datasets/ML_FB/user_likes_train.pkl', 'wb') as f:
    pickle.dump(user_likes_train, f) 
with open('datasets/ML_FB/user_likes_test.pkl', 'wb') as f:
    pickle.dump(user_likes_test, f) 
