import os, pickle, sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LoadDataset(Dataset):
    # mode is 'test' or 'train'
    def __init__(self, dataset_name, mode, neg_ratio, noise):
        self.name = dataset_name
        self.neg_ratio = int(neg_ratio)
        assert neg_ratio >= 1
        self.noise = noise

        # load datasets and info mappings
        path = os.path.join('datasets', self.name)
        
        rec = np.load(os.path.join(path, 'rec.npy'), allow_pickle=True)
        if mode == 'train': 
            self.rec_train = np.load(os.path.join(path, 'rec_train.npy'), allow_pickle=True)
            self.kg = np.load(os.path.join(path, 'kg.npy'), allow_pickle=True)
            self.data = np.concatenate((self.rec_train, self.kg), axis=0) # must shuffle dataloader!

        elif mode == 'test':
            self.data = np.load(os.path.join(path, 'rec_test.npy'), allow_pickle=True)
            
            # info about users, items and total kg items
            self.users = np.unique(rec[:,0])
            self.rec_items = np.unique(rec[:,2])
            
            with open(os.path.join(path, 'user_likes_map.pkl'), 'rb') as f:
                self.user_likes_map = pickle.load(f)
        
        # total num of items in kg + rec
        self.num_items = max(np.max(self.kg), np.max(rec))
        self.num_rel = max(np.max(self.kg[:,1]), np.max(rec[:,1]))
        
        # for self.print_triplet 
        with open(os.path.join(path, 'item_map.pkl'), 'rb') as f:
            self.item_map = pickle.load(f)
        with open(os.path.join(path, 'rel_map.pkl'), 'rb') as f:
            self.link_map = pickle.load(f)

    def __getitem__(self, index):
        pos = self.data[index]
        neg = self.get_negatives(pos)
        data = np.concatenate((np.expand_dims(pos, axis=0), neg), axis=0)
        
        # add label information in col 4
        labels = -np.ones((data.shape[0], 1))
        labels[0] = 1
        data = np.hstack((data, labels))
        return torch.from_numpy(data).long()

    def __len__(self):
        return self.data.shape[0]

    # negative sampling
    # TODO: update this with better policy
    def get_negatives(self, pos):
        neg = np.ones((self.neg_ratio, 3), dtype=np.int32)
        neg[:] *= pos
        for i in range(self.neg_ratio):
            ind = 0 if np.random.rand() < 0.5 else 2 # flip head or tail
            neg[i, ind] = self.replace_item(pos[ind])
        return neg

    def replace_item(self, item):
        while True:
            sample = np.random.randint(self.num_items)
            if sample != item: break
        return sample

    # prints human-readable triplets from rec or kg triplets
    def print_triplet(self, triplet):
        # subject 
        if triplet[0] in self.item_map:
            s = self.item_map[triplet[0]]
        else:
            s = 'Freebase ID: {}'.format(triplet[0])
        # relationship
        r = self.link_map[triplet[1]]
        # object
        if triplet[2] in self.item_map:
            o = self.item_map[triplet[2]]
        else:
            o = 'Freebase ID: {}'.format(triplet[2])
        print('<{} -- {} -- {}>'.format(s,r,o))

#dset = LoadDataset('ML_FB','train', 2, 0.2)
#dataloader = DataLoader(dset, batch_size=5, shuffle=True, num_workers=4)

#import torch.nn as nn
#device = 'cpu'
#item_emb = nn.Embedding(dset.num_items, 8).to(device)
#rel_emb = nn.Embedding(dset.num_rel+1, 8).to(device)
#print(dset.num_rel)

#for i, x in enumerate(dataloader): 
    #x = x.to(device)

    #h = item_emb(x[:,:,0])
    #r = rel_emb(x[:,:,1])
    #t = item_emb(x[:,:,2])
    #print(h)
    #break