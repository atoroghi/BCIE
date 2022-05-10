import os, pickle, sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LoadDataset(Dataset):
    # mode is 'test' or 'train'
    def __init__(self, mode, args, power=3/4):
        self.name = args.dataset
        self.neg_ratio = int(args.neg_ratio)
        assert self.neg_ratio >= 1

        self.noise = args.ni
        self.workers = args.workers
        self.batch_size = args.batch_size
        self.par_batch = self.batch_size // self.workers

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
        # TODO: make this loss data specific (this is dependent of data format)
        self.users = np.unique(rec[:,0])
        self.rec_items = np.unique(rec[:,2])
        self.last_index = self.data.shape[0] // self.par_batch
        
        self.num_items = np.max(rec)
        self.num_rel = rec[0,1]

        with open(os.path.join(path, 'user_likes_map.pkl'), 'rb') as f:
            self.user_likes_map = pickle.load(f)

        # preprocessing for negative sampling

        
        # for self.print_triplet 
        with open(os.path.join(path, 'item_map.pkl'), 'rb') as f:
            self.item_map = pickle.load(f)
        with open(os.path.join(path, 'rel_map.pkl'), 'rb') as f:
            self.link_map = pickle.load(f)

    def __getitem__(self, index):
        if index != self.last_index:
            pos = self.data[index : index + self.par_batch]
        else:
            pos = self.data[index :]
        neg = self.get_negatives(pos)
        
        data = np.concatenate((pos, neg), axis=0)
        
        # add label information in col 4
        labels = -np.ones((data.shape[0], 1))
        labels[:pos.shape[0]] = 1
        data = np.hstack((data, labels))

        return torch.from_numpy(data).long()

    def __len__(self):
        # each worker selects (batch / num workers)
        return self.data.shape[0] // (self.par_batch)

    # negative sampling
    # TODO: make this parallel?
    def get_negatives(self, pos):
        for i in range(pos.shape[0]):
            neg = np.ones((self.neg_ratio, 3), dtype=np.int32)
            neg[:] *= pos[i]
            
            for j in range(self.neg_ratio):
                ind = 0 if np.random.rand() < 0.5 else 2 # flip head or tail
                neg[j, ind] = self.replace_item(pos[i, ind])

            if i == 0:
                final_neg = neg
            else:
                final_neg = np.concatenate((final_neg, neg), axis=0)
        return final_neg

    def replace_item(self, item):
        while True:
            # fast for uniform sample
            if self.power == 0:
                sample = np.random.randint(self.num_items)
            # discrete inverse sampling
            else:
                print()

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