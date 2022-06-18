import os, pickle, sys, time
import torch
import numpy as np

class DataLoader:
    def __init__(self, args):
        self.name = args.dataset
        self.neg_ratio = int(args.neg_ratio)
        self.sample_type = args.sample_type
        self.batch_size = args.batch_size

        # load datasets and info mappings
        path = os.path.join('datasets', self.name)
        self.rec_train = np.load(os.path.join(path, 'rec_train.npy'), allow_pickle=True)

        # info about users etc
        self.rec_test = np.load(os.path.join(path, 'rec_test.npy'), allow_pickle=True)
        rec = np.concatenate((self.rec_train, self.rec_test))
        self.users = np.unique(rec[:,0])
        self.test_users = np.unique(rec[:,0])
        self.items = np.unique(rec[:,2])
        self.likes_link = 0 # hard coded
        self.first_userid = np.min(self.users) # used for printing triplets        
        
        # load data for training
        if args.kg == 'kg':
            self.kg = np.load(os.path.join(path, 'kg.npy'), allow_pickle=True)
            self.data = np.concatenate((self.rec_train, self.kg))
            self.num_item = np.max(self.data) + 1
            self.num_rel = np.max(self.kg[:,1]) + 1
        elif args.kg == 'no_kg':
            self.data = self.rec_train
            self.num_item = np.max(rec) + 1
            self.num_rel = 1
        else:
            print('not valid: ', args.kg)
            sys.exit()

        self.n_batches = int(np.ceil(self.data.shape[0] / args.batch_size))

        # user likes map for testing
        with open(os.path.join(path, 'user_likes_map.pkl'), 'rb') as f:
            self.user_likes_map = pickle.load(f)       

        # class for negative sampling
        if self.sample_type == 'single':
            self.sampler = SingleSample(self.data, power=args.neg_power)
        elif self.sample_type == 'double':
            self.sampler = DoubleSample(self.data, power=args.neg_power)

        # load data for printing relation
        with open(os.path.join(path, 'id2html.pkl'), 'rb') as f:
            self.item_map = pickle.load(f)
        with open(os.path.join(path, 'rel_map.pkl'), 'rb') as f:
            self.rel_map = pickle.load(f)

    def print_triple(self, triple):
        try:
            head = self.item_map[triple[0]]
        except:
            head = 'User {}'.format(triple[0] - self.first_userid)
        rel = self.rel_map[triple[1]]
        tail = self.item_map[triple[2]]
        print('{}, {}, {}'.format(head, rel, tail))

    def shuffle(self):
        self.data = np.random.permutation(self.data)

    def get_batch(self, i):
        if i != self.n_batches - 1:
            pos = self.data[i * self.batch_size : (i + 1) * self.batch_size]
        else:
            pos = self.data[i * self.batch_size : ]

        neg = self.get_negatives(pos)
        data = np.concatenate((pos, neg), axis=0)
        
        # add label information in col 4
        labels = -np.ones((data.shape[0], 1))
        labels[:pos.shape[0]] = 1
        data = np.hstack((data, labels))

        return torch.from_numpy(data).long()

    # negative sampling
    def get_negatives(self, pos):
        n = self.neg_ratio * pos.shape[0] # number of neg samples
        neg = np.repeat(np.copy(pos), self.neg_ratio, axis=0)
        
        mask = np.random.randint(0, 2, size=(n))
        mask = np.vstack((mask, np.ones(n), 1 - mask)).T

        if self.sample_type == 'single':
            samples = self.sampler.sample(n)
            samples = np.vstack((samples, np.zeros(n), samples)).T
        elif self.sample_type == 'double':
            head_samples, tail_samples = self.sampler.sample(n)
            samples = np.vstack((tail_samples, np.zeros(n), head_samples)).T

        neg = neg * mask + samples * (1 - mask)
        return neg

# treats head and tail as single dist
class SingleSample:
    def __init__(self, data, power=0):
        #assert power >= 0 and power <= 1
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
        #assert power >= 0 and power <= 1
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
            head_sample = np.random.randint(self.num_items_head + 1, size=(n))
            tail_sample = np.random.randint(self.num_items_tail + 1, size=(n))
        
        # discrete inverse sampling    
        mask = np.random.randint(0, 2, size=(n))
        mask = np.vstack((mask, np.ones(n), 1 - mask)).T

        if self.sample_type == 'single':
            samples = self.sampler.sample(n)
            samples = np.vstack((samples, np.zeros(n), samples)).T
        elif self.sample_type == 'double':
            head_samples, tail_samples = self.sampler.sample(n)
            samples = np.vstack((tail_samples, np.zeros(n), head_samples)).T

        neg = neg * mask + samples * (1 - mask)
        return neg