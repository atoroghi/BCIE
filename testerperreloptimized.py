import torch
import os, sys
import numpy as np
from measureperrel import Measureperrel
from os import listdir
from os.path import isfile, join
import pickle
import copy
from tqdm import tqdm


def Testerperreloptimized(dataset,model,args,epoch):
        # to store embeddings and projected embeddings (head and tail)
        device = torch.device('cpu')
        num_ent=dataset.num_ent
        num_rel=dataset.num_rel
        ents=dataset.ents
        rels=dataset.rels
        facts_count={}
        path=os.path.join('datasets', dataset.name)
        #results_path = 'results/perrels/'+args.test_name
        measure = Measureperrel('results/perrels/'+args.test_name+str(epoch)+'/')
        with open(path+'/heads_filtering.pkl', 'rb') as f:
            heads_filtering = pickle.load(f)
        with open(path+'/tails_filtering.pkl', 'rb') as f:
            tails_filtering = pickle.load(f)
        ent_h = np.zeros([num_ent, args.emb_dim])
        ent_t = np.zeros([num_ent, args.emb_dim])
        rels_r = np.zeros([num_rel, args.emb_dim])
        rels_r_inv = np.zeros([num_rel, args.emb_dim])
        ent2index = dict(zip(ents, list(range(0, num_ent))))
        rel2index = dict(zip(rels,list(range(0, num_rel))))
        #ent2indexinv = dict(zip(list(range(0, num_ent)),ents))
        #rel2indexinv = dict(zip(list(range(0, num_rel)),rels))

        for ent in ents:
            index=ent2index[ent]
            ent_tensor = torch.tensor([ent]).long()
            ent_h[index]= model.ent_h_embs(ent_tensor).detach().numpy()
            ent_t[index]= model.ent_t_embs(ent_tensor).detach().numpy()
        for rel in rels:
            index=rel2index[rel]
            rel_tensor = torch.tensor([rel]).long()
            rels_r[index] = model.rel_embs(rel_tensor).detach().numpy()
            rels_r_inv[index] = model.rel_inv_embs(rel_tensor).detach().numpy()
        
        for rel in range(0,num_rel):
            facts_count[rel]=0
        
        for i, fact in tqdm(enumerate(dataset.data_test)):
            _,rel_num,_=fact
            facts_count[rel_num]+=1
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
                        mask_head=np.in1d(ranked_head,to_remove_indices,invert=True)
                        ranked_head_new=ranked_head[mask_head]
                    rank = int(np.where(ranked_head_new == h)[0])+1
                    measure.update(rank, rel_num)
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
                        to_remove_indices=[ent2index[x] for x in to_remove if x != t_ent]
                        mask_tail=np.in1d(ranked_tail,to_remove_indices,invert=True)
                        ranked_tail_new = ranked_tail[mask_tail]
                    rank = int(np.where(ranked_tail_new == t)[0]) +1
                    measure.update(rank,rel_num)

        for rel_num in range(0,num_rel):
            measure.normalize(rel_num,facts_count[rel_num])
        #hit1,hit3,hit10,mr,mrr=measure.print_()
        hit1,hit3,hit10,mr,mrr=measure.pass_back()
        return hit1,hit3,hit10,mr,mrr

