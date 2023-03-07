import os, pickle, sys, time
import torch
import numpy as np
import random

class DataLoader:
    def __init__(self, args):
        self.name = args.dataset
        self.neg_ratio = int(args.neg_ratio)
        self.sample_type = args.sample_type
        self.batch_size = args.batch_size
        self.fold = args.fold
        self.type_checking = args.type_checking 

        # load datasets and info mappings
        path = os.path.join('datasets', self.name, 'fold {}'.format(self.fold))
        main_path = os.path.join('datasets', self.name)

        self.rec_train = np.load(os.path.join(path, 'train.npy'), allow_pickle=True)
        self.rec_test = np.load(os.path.join(path, 'test.npy'), allow_pickle=True)
        self.rec_val = np.load(os.path.join(path, 'val.npy'), allow_pickle=True)
        #print("num_items")
        #print(np.unique(self.rec_train[:,2]).shape)
        #print("num_users")
        #print(np.unique(self.rec_train[:,0]).shape)

        # load data for training
        if args.kg == 'kg':
            self.kg = np.load(os.path.join('datasets', self.name, 'kg.npy'), allow_pickle=True)        
            self.train_data = np.concatenate((self.rec_train, self.kg))    
            self.num_rel = np.max(self.kg[:,1]) + 1
            self.all_rel = np.unique(self.kg[:,1])
        elif args.kg == 'no_kg':
            #self.kg_train = np.load(os.path.join(path, 'kg_train.npy'), allow_pickle=True)        
            #self.kg_test = np.load(os.path.join(path, 'kg_test.npy'), allow_pickle=True)        
            #self.kg = np.concatenate((self.kg_train, self.kg_test))

            self.train_data = self.rec_train
            self.num_rel = 1
        else:
            print('kg mode not valid')
            sys.exit()

        # useful information about data
        self.num_item = np.max(self.train_data) + 1
        self.n_batches = int(np.ceil(self.train_data.shape[0] / args.batch_size))
        self.likes_link = 0 # hard coded
        #self.first_userid = np.min(self.users) # used for printing triplets

        # class for negative sampling
        if self.type_checking == 'no':
            if self.sample_type == 'combo':
                self.sampler = SingleSample(self.train_data, power=args.neg_power)
            elif 'split' in self.sample_type:
                self.sampler = DoubleSample(self.train_data, power=args.neg_power)


        if self.type_checking == 'check':
            with open(os.path.join(main_path, 'valid_heads.pkl'), 'rb') as f:
                self.valid_heads = pickle.load(f)
            with open(os.path.join(main_path, 'valid_tails.pkl'), 'rb') as f:
                self.valid_tails = pickle.load(f)
            with open(os.path.join(main_path, 'valid_heads_freq.pkl'), 'rb') as f:
                self.valid_heads_freq = pickle.load(f)
            with open(os.path.join(main_path, 'valid_tails_freq.pkl'), 'rb') as f:
                self.valid_tails_freq = pickle.load(f)


            if self.sample_type == 'combo':
                print("combo sample type not supported with type checking")
                sys.exit()
            elif 'split' in self.sample_type:
                self.sampler = DoubleSampleType(self.valid_heads, self.valid_tails, self.valid_heads_freq,
                self.valid_tails_freq, power=args.neg_power)
            
        # user likes
        #with open(os.path.join(path, 'user_likes_test.pkl'), 'rb') as f:
        #    self.user_likes_map = pickle.load(f)

        #with open(os.path.join(path, 'user_likes_whole.pkl'), 'rb') as f:
        #    self.user_likes_whole = pickle.load(f)

        # load data for printing relation
        #with open(os.path.join(path, 'id2html.pkl'), 'rb') as f:
        #    self.item_map = pickle.load(f)
        #with open(os.path.join(path, 'rel_map.pkl'), 'rb') as f:
        #    self.rel_map = pickle.load(f)

    def print_triple(self, triple):
        try:
            head = self.item_map[triple[0]]
        except:
            head = 'User {}'.format(triple[0] - self.first_userid)
        rel = self.rel_map[triple[1]]
        tail = self.item_map[triple[2]]
        print('{}, {}, {}'.format(head, rel, tail))

    def shuffle(self):
        self.train_data = np.random.permutation(self.train_data)

    def get_batch(self, i):
        if i != self.n_batches - 1:
            pos = self.train_data[i * self.batch_size : (i + 1) * self.batch_size]
        else:
            pos = self.train_data[i * self.batch_size : ]

        neg = self.get_negatives(pos)
        data = np.vstack((pos, neg))
        
        # add label information in col 4
        labels = -np.ones((data.shape[0], 1))
        labels[:pos.shape[0]] = 1
        data = np.hstack((data, labels))

        return torch.from_numpy(data).long()

    # negative sampling
    def get_negatives(self, pos):
        n = self.neg_ratio * pos.shape[0] # number of neg samples
        pos = np.repeat(np.copy(pos), self.neg_ratio, axis=0)
        # TODO: switch to only modify tail (to make similar to svd)
        mask = np.random.randint(0, 2, size=(n))
        mask = np.vstack((mask, np.ones(n), 1 - mask)).T

        if self.type_checking == 'check':
            head_samples, tail_samples = self.sampler.sample(pos,self.neg_ratio)
            if 'reg' in self.sample_type:
                samples = np.vstack((head_samples, np.zeros(n), tail_samples)).T
            elif 'rev' in self.sample_type:
                samples = np.vstack((tail_samples, np.zeros(n), head_samples)).T

        if self.type_checking == 'no':
            if self.sample_type == 'combo':
                samples = self.sampler.sample(n)
                samples = np.vstack((samples, np.zeros(n), samples)).T
            # soft samples work better rn?
            elif 'split' in self.sample_type:
                # option to flip type
                head_samples, tail_samples = self.sampler.sample(n)
                
                if 'reg' in self.sample_type:
                    samples = np.vstack((head_samples, np.zeros(n), tail_samples)).T
                elif 'rev' in self.sample_type:
                    samples = np.vstack((tail_samples, np.zeros(n), head_samples)).T

        neg = pos * mask + samples * (1 - mask)
        return neg

# TODO: merge these into a single class
# treats head and tail as single dist
class SingleSample:
    def __init__(self, data, power=0):
        self.power = power
        self.total = np.concatenate((data[:,0], data[:,2]))
        self.num_items = np.max(self.total)

        # otherwise uniform sampling
        if self.power != 0:
            self.dist = np.zeros(self.num_items + 1)

            for i in range(self.total.shape[0]):
                index = self.total[i]
                self.dist[index] += 1

            self.dist = np.power(self.dist, self.power)
            self.dist = self.dist / np.sum(self.dist)
            self.dist = self.dist.astype(np.double)
    
    def sample(self, n):
        # efficient uniform sampling
        if self.power == 0:
            sample = np.random.randint(self.num_items + 1, size=(n))
        # discrete inverse sampling
        else:
            sample = np.random.choice(self.num_items + 1, size=(n), p=self.dist)

        return sample

# looks at head and tail seperately
class DoubleSample:
    def __init__(self, data, power=0):
        self.power = power
        self.head = data[:,0]
        self.tail = data[:,2]
        self.num_items_head = np.max(self.head)
        self.num_items_tail = np.max(self.tail)

        # otherwise uniform sampling
        if self.power != 0:
            self.head_dist = np.zeros(self.num_items_head + 1)
            self.tail_dist = np.zeros(self.num_items_tail + 1)

            for i in range(self.head.shape[0]):
                index = self.head[i]
                self.head_dist[index] += 1

            for i in range(self.tail.shape[0]):
                index = self.tail[i]
                self.tail_dist[index] += 1

            self.head_dist = np.power(self.head_dist, self.power)
            self.head_dist = self.head_dist / np.sum(self.head_dist)
            self.head_dist = self.head_dist.astype(np.double)

            self.tail_dist = np.power(self.tail_dist, self.power)
            self.tail_dist = self.tail_dist / np.sum(self.tail_dist)
            self.tail_dist = self.tail_dist.astype(np.double)

    def sample(self, n):
        # efficient uniform sampling
        if self.power == 0:
            head_samples = np.random.randint(self.num_items_head + 1, size=(n))
            tail_samples = np.random.randint(self.num_items_tail + 1, size=(n))
        # discrete inverse sampling
        else:
            head_samples = np.random.choice(self.num_items_head + 1, size=(n), p=self.head_dist)
            tail_samples = np.random.choice(self.num_items_tail + 1, size=(n), p=self.tail_dist)
        return head_samples, tail_samples

class DoubleSampleType:
    def __init__(self, valid_heads, valid_tails, valid_heads_freq, valid_tails_freq, power=0):
        self.power = power
        self.valid_heads = valid_heads
        self.valid_tails = valid_tails
        self.valid_heads_freq = valid_heads_freq
        self.valid_tails_freq = valid_tails_freq

    def sample(self, pos, neg_ratio):
        head_samples = pos[:,0]
        rels = pos[:,1]
        tail_samples = pos[:,2]
        rel_unqs, rel_freqs = np.unique(rels,return_counts=True) 
        if self.power == 0:
            for rel , freq in zip(rel_unqs, rel_freqs):
                rel_inds = np.where((rels == rel))
                heads_chosen = np.random.choice(self.valid_heads[rel], size=(freq))
                tails_chosen = np.random.choice(self.valid_tails[rel], size=(freq))
                head_samples[rel_inds] = heads_chosen
                tail_samples[rel_inds] = tails_chosen
        else:
            for rel , freq in zip(rel_unqs, rel_freqs):
                rel_inds = np.where((rels == rel))
                head_probs = self.valid_heads_freq[rel]
                tail_probs = self.valid_tails_freq[rel]
                heads_chosen = np.random.choice(self.valid_heads[rel], size=(freq), p=head_probs)
                tails_chosen = np.random.choice(self.valid_tails[rel], size=(freq), p=tail_probs)
                head_samples[rel_inds] = heads_chosen
                tail_samples[rel_inds] = tails_chosen
        return head_samples, tail_samples