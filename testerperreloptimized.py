import torch
import os, sys
import numpy as np
from dataset import Dataset
from measureperrel import Measureperrel
from os import listdir
from os.path import isfile, join
import pickle
import copy
from tqdm import tqdm

class Testerperreloptimized:
    def __init__(self, dataset, model_path, valid_or_test,args):
        self.device = torch.device('cpu')
        self.model = torch.load(model_path, map_location = self.device)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.path = os.path.join('datasets', self.dataset.name)
        self.measure = Measureperrel(self.path)
        #self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())
        self.facts_count={}
        self.num_ent=self.dataset.num_ent
        self.num_rel=self.dataset.num_rel
        self.ents=self.dataset.ents
        self.rels=self.dataset.rels
        self.args=args
    def test(self):
        # to store embeddings and projected embeddings (head and tail)
        with open(self.path+'/heads_filtering.pkl', 'rb') as f:
            heads_filtering = pickle.load(f)
        with open(self.path+'/tails_filtering.pkl', 'rb') as f:
            tails_filtering = pickle.load(f)
        ent_h = np.zeros([self.num_ent, self.args.emb_dim])
        ent_t = np.zeros([self.num_ent, self.args.emb_dim])
        rels_r = np.zeros([self.num_rel, self.args.emb_dim])
        rels_r_inv = np.zeros([self.num_rel, self.args.emb_dim])
        ent2index = dict(zip(self.ents, list(range(0, self.num_ent))))
        rel2index = dict(zip(self.rels,list(range(0, self.num_rel))))
        ent2indexinv = dict(zip(list(range(0, self.num_ent)),self.ents))
        rel2indexinv = dict(zip(list(range(0, self.num_rel)),self.rels))

        for ent in self.ents:
            index=ent2index[ent]
            ent_tensor = torch.tensor([ent]).long()
            ent_h[index]= self.model.ent_h_embs(ent_tensor).detach().numpy()
            ent_t[index]= self.model.ent_t_embs(ent_tensor).detach().numpy()
        for rel in self.rels:
            index=rel2index[rel]
            rel_tensor = torch.tensor([rel]).long()
            rels_r[index] = self.model.rel_embs(rel_tensor).detach().numpy()
            rels_r_inv[index] = self.model.rel_inv_embs(rel_tensor).detach().numpy()
        
        for rel_num in range(0,self.num_rel):
            self.facts_count[rel_num]=0
        
        for i, fact in tqdm(enumerate(self.dataset.data_test)):
            raw_or_fil="fil"
            _,rel_num,_=fact
            self.facts_count[rel_num]+=1
            h_ent,r_rel,t_ent=fact
            h=ent2index[h_ent]
            r=rel2index[r_rel]
            t=ent2index[t_ent]
            for head_or_tail in ["head", "tail"]:
                if head_or_tail=="head": #?,r,t
                    tail_proj = np.multiply(rels_r[r],ent_t[t])
                    for_prod = np.multiply(ent_h,tail_proj)
                    head_proj = np.multiply(rels_r_inv[r],ent_h[t])
                    inv_prod = np.multiply(ent_t,head_proj)
                    scores = np.clip(np.sum(for_prod + inv_prod, axis=1), -40, 40)
                    ranked_head = scores.argsort()[::-1]
                    ranked_head_new = 1*ranked_head
                    #remove other correct heads from competition
                    if tuple(fact[1:]) in heads_filtering.keys():
                        to_remove=heads_filtering[tuple(fact[1:])]
                        to_remove_indices=[ent2index[x] for x in to_remove if x != h_ent]
                        #dims = np.maximum(to_remove_indices.max(0),ranked_head.max(0))+1
                        #ranked_head_new = ranked_head[~np.in1d(np.ravel_multi_index(ranked_head.T,dims),np.ravel_multi_index(to_remove_indices.T,dims))]
                        #ranked_head_new=np.array([x for x in ranked_head if x not in to_remove_indices])
                        mask_head=np.in1d(ranked_head,to_remove_indices,invert=True)
                        ranked_head_new=ranked_head[mask_head]
                    rank = int(np.where(ranked_head_new == h)[0])+1
                    self.measure.update(rank, raw_or_fil,rel_num)
                if head_or_tail=="tail": #h,r,?
                    head_proj = np.multiply(rels_r[r],ent_h[h])
                    for_prod = np.multiply(ent_t,head_proj)
                    tail_proj = np.multiply(ent_t[h],rels_r_inv[r])
                    inv_prod = np.multiply(ent_h,tail_proj)
                    scores = np.clip(np.sum(for_prod + inv_prod, axis=1), -40, 40)
                    ranked_tail = scores.argsort()[::-1]
                    ranked_tail_new = 1*ranked_tail

                    #remove other correct heads from competition
                    if tuple(fact[:2]) in tails_filtering.keys():
                        to_remove=tails_filtering[tuple(fact[:2])]
                        to_remove_alaki=[ent2index[x] for x in to_remove]
                        to_remove_indices=[ent2index[x] for x in to_remove if x != t_ent]
                        #ranked_tail_new=np.array([x for x in ranked_tail if x not in to_remove_indices])
                        mask_tail=np.in1d(ranked_tail,to_remove_indices,invert=True)
                        ranked_tail_new = ranked_tail[mask_tail]
                    rank = int(np.where(ranked_tail_new == t)[0]) +1
                    self.measure.update(rank, raw_or_fil,rel_num)

        for rel_num in range(0,self.num_rel):
            self.measure.normalize(rel_num,self.facts_count[rel_num])
        self.measure.print_()
        return self.measure.mrr["fil"]

