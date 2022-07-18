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
        self.likelihood = args.likelihood

        # load datasets and info mappings
        path = os.path.join('datasets', self.name)
        self.rec_train = np.load(os.path.join(path, 'rec_train.npy'), allow_pickle=True)
        # info about users etc
        self.rec_test = np.load(os.path.join(path, 'rec_test.npy'), allow_pickle=True)
        rec = np.concatenate((self.rec_train, self.rec_test))
        self.users = np.unique(self.rec_train[:,0])
        self.test_users = np.unique(self.rec_test[:,0])
        self.items = np.unique(self.rec_train[:,2])
        _ , cnt = np.unique(np.concatenate([self.items,self.rec_train[:,2]]),return_counts=True)
        self.freqs = cnt[0:len(self.items)]
        self.freqs = self.freqs / np.sum(self.freqs)
        self.likes_link = 0 # hard coded
        self.first_userid = np.min(self.users) # used for printing triplets
        self.kg = np.load(os.path.join(path, 'kg.npy'), allow_pickle=True)        
        # load data for training
        if args.kg_inclusion=='True':
            
            self.kg_train = self.kg[:610317]
            self.kg_test = self.kg[610317:612317]
            self.data = np.concatenate((self.rec_train, self.kg_train))
            #self.data_test=np.concatenate((self.rec_test[100000:101000],self.kg_test))
            self.data_test=self.rec_test[:10000]
            self.num_item = np.max(self.data) + 1
            self.num_rel = np.max(self.kg[:,1]) + 1
        elif args.kg_inclusion == 'False':
            self.data = self.rec_train
            self.data_test=self.rec_test[:10000]
            self.num_item = np.max(rec) + 1
            self.num_rel = 1
        else:
            print('not valid: ', args.kg)
            sys.exit()
        all_ents=np.concatenate(((np.delete(self.data,1,axis=1),np.delete(self.kg[610317:],1,axis=1),np.delete(self.rec_test,1,axis=1))))
        self.ents=np.unique(all_ents)
        self.num_ent = len(self.ents)
        self.rels=np.unique(np.concatenate((self.data[:,1],self.rec_test[:,1],self.kg[:,1])))
        rec = np.load(os.path.join(path, 'rec.npy'), allow_pickle=True)
        self.items = np.unique(self.rec_train[:,2])
        self.max_item = np.max(self.data) + 1
        self.likes_link = np.max(self.rec_train[:,1])

        self.n_batches = int(np.ceil(self.data.shape[0] / args.batch_size))

        # user likes map for testing
        with open(os.path.join(path, 'user_likes_mine.pkl'), 'rb') as f:
            self.user_likes_map = pickle.load(f)       
        with open(os.path.join(path, 'user_likes_mine_whole.pkl'), 'rb') as f:
            self.user_likes_mine_whole = pickle.load(f)
        with open(os.path.join(path, 'user_likes_mine_train.pkl'), 'rb') as f:
            self.user_likes_train = pickle.load(f)

        # user likes train for generating negative samples in the MF style 
        # (every items that we haven't observed user's interest, we call it a neg example)   

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
        
        if self.likelihood == 'logistic':
            neg = self.get_negatives(pos)

        elif self.likelihood == 'gaussian':
            neg = self.get_negatives_mf(pos)
        if neg.shape[0]>0:
            data = np.concatenate((pos, neg), axis=0)
        else:
            data = pos
        
        
        # add label information in col 4
        labels = -np.ones((data.shape[0], 1))
        labels[:pos.shape[0]] = 1
        data = np.hstack((data, labels))

        #return torch.from_numpy(data).long()
        return torch.as_tensor(data).long()

    # negative sampling the new style
    # previously, we were doing negative sampling by changing either head or tail of the (user,likes,item) triple
    # in WRMF or SVD though, we have a sparse matrix in which every non-observed interaction is deemed as a negative interation (example)
    # this was what Scott told me to change in our neg sampling but adding all these facts to the rec matrix makes it insanely large (although now I 
    # think I should change the code to use a sparse matrix and I am trying that now)
    # What I'm doing here is for each positive example, select an item randomly (with probabilities proportionate to frequency) from the ones the user
    #  hasn't interacted with and add it as a negative example. This way we don't cover all negative examples, but use a number of them in training.
    #  

    def get_negatives_mf(self,pos):
        neg_samples = []
        #for example in pos:
        #    dislikes = list(set(self.items)-set(self.user_likes_train[example[0]]))
        #    neg_items = random.sample(dislikes, self.neg_ratio)
        #    for item in neg_items:
        #        neg_samples.append([example[0],0,item])
        for example in pos:
            neg=[]
            while len(neg)<self.neg_ratio:
                dis = (np.random.choice(self.items, size = 1, p= self.freqs)[0])
                #dis = random.choice(self.items)
                if dis not in self.user_likes_train[example[0]]:
                    neg.append([example[0],0,dis])
            neg_samples += neg
        
        return np.array(neg_samples)



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

