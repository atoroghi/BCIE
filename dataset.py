import numpy as np
import random
import torch
import math

class Dataset:
    def __init__(self, ds_name, noise):
        self.name = ds_name
        self.dir = "datasets/" + ds_name + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.items = []
        self.users = []
        self.users_likes = {}
        self.noise_intensity = noise
        self.data = {spl: self.read(self.dir + spl + ".txt") for spl in ["train", "valid", "test"]}
        self.batch_index = 0

    def read(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        triples = np.zeros((len(lines), 3),dtype=int)
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids(line.strip().split("\t")),dtype=int)

            if triples[i][1] == 0:
                if triples[i][0] not in self.users:
                    self.users.append(triples[i][0])
                if triples[i][0] not in self.users_likes:
                    self.users_likes[triples[i][0]]=[]
                else:
                    if triples[i][2] not in self.users_likes[triples[i][0]]:
                        self.users_likes[triples[i][0]].append(triples[i][2])
                    
                if triples[i][2] not in self.items:
                    self.items.append(triples[i][2])
        noise_intensity=self.noise_intensity
        num_facts=np.arange(triples.shape[0])
        to_change=np.random.choice(num_facts,int(np.ceil(noise_intensity*triples.shape[0])),replace=False)
        for ind in to_change:
            ent=np.random.choice([0,2],1)
            triples[ind][ent]=triples[np.random.choice(num_facts[:-2],1)[0]][ent]

        return triples

    def num_ent(self):
        return len(self.ent2id)
    
    def num_rel(self):
        return len(self.rel2id)
                     
    def triple2ids(self, triple):
        ret = [self.get_ent_id(triple[0]), self.get_rel_id(triple[1]), self.get_ent_id(triple[2])]
        return [self.get_ent_id(triple[0]), self.get_rel_id(triple[1]), self.get_ent_id(triple[2])]
                     
    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]
            
    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]
                     
    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent
                     
    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]):
            batch = self.data["train"][self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            self.batch_index = 0
        return np.append(batch, np.ones((len(batch), 1)), axis=1).astype("int") #appending the +1 label
                     
    def generate_neg(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if random.random() < 0.5:
                neg_batch[i][0] = self.rand_ent_except(neg_batch[i][0]) #flipping head
            else:
                neg_batch[i][2] = self.rand_ent_except(neg_batch[i][2]) #flipping tail
        neg_batch[:,-1] = -1
        return neg_batch

    def next_batch(self, batch_size, neg_ratio, device):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch = self.generate_neg(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        np.random.shuffle(batch)
        heads  = torch.tensor(batch[:,0]).long().to(device)
        rels   = torch.tensor(batch[:,1]).long().to(device)
        tails  = torch.tensor(batch[:,2]).long().to(device)
        labels = torch.tensor(batch[:,3]).float().to(device)
        return heads, rels, tails, labels
    
    def was_last_batch(self):
        return (self.batch_index == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))


