import torch
from dataset import Dataset
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from dataload import LoadDataset
from tqdm import tqdm
from recommender import Recommender
import wandb


class Tester:
    def __init__(self, model_path, valid_or_test, dataset, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location = 'cpu')
        self.model.eval()
        self.valid_or_test = valid_or_test
        #self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())
        self.emb_dim=args.emb_dim
        self.model_path=model_path
        self.dataset=dataset
        self.items_list=dataset.rec_items
        self.users_likes=dataset.user_likes_map
        #self.users_list=self.users_likes.keys()
        self.users_list=list(self.users_likes.keys())
        self.items_embeddings_head=np.zeros([len(self.items_list),self.emb_dim])
        self.items_embeddings_tail=np.zeros([len(self.items_list),self.emb_dim])
        self.items_index=dict(zip(self.items_list,list(range(0,len(self.items_list)))))
        self.items_index_inverse=dict(zip(list(range(0,len(self.items_list))),self.items_list))
        self.users_embeddings_head_proj=np.zeros([np.max(self.users_list)-self.users_list[0]+1,self.emb_dim])
        self.users_embeddings_tail_proj=np.zeros([np.max(self.users_list)-self.users_list[0]+1,self.emb_dim])
        #os.environ['WANDB_API_KEY']='d606ae06659873794e6a1a5fb4f55ffc72aac5a1'
        #os.environ['WANDB_USERNAME']='atoroghi'
        #wandb.login(key="d606ae06659873794e6a1a5fb4f55ffc72aac5a1")
        #wandb.init(project="pre-critiquing",config={"lr": 0.1},settings=wandb.Settings(start_method="fork"))
        wandb.init(project="pre-critiquing",config={"lr": 0.1})
        #os.environ['WANDB_API_KEY']='d606ae06659873794e6a1a5fb4f55ffc72aac5a1'
        #os.environ['WANDB_USERNAME']='atoroghi'
        wandb.config.update(args,allow_val_change=True)

        for item in self.items_list:
            t = torch.tensor([item]).long()
            item_embedding_head = self.model.ent_h_embs(t)
            item_embedding_tail = self.model.ent_t_embs(t)
            to_head=(item_embedding_head.detach().numpy()).reshape((1,self.emb_dim))
            to_tail=(item_embedding_tail.detach().numpy()).reshape((1,self.emb_dim))
            index=self.items_index[item]
            self.items_embeddings_head[index]=to_head
            self.items_embeddings_tail[index]=to_tail
        for user in self.users_list:
            h = torch.tensor([user]).long()
            r=torch.tensor([47]).long()
            users_embedding_head = self.model.ent_h_embs(h)
            users_embedding_tail = self.model.ent_t_embs(h)
            likes_embedding = self.model.rel_embs(r)
            likes_embedding_inv = self.model.rel_inv_embs(r)
            self.users_embeddings_head_proj[user-self.users_list[0]]=np.multiply((users_embedding_head.detach().numpy()),(likes_embedding.detach().numpy()))
            self.users_embeddings_tail_proj[user-self.users_list[0]]=np.multiply((users_embedding_tail.detach().numpy()),(likes_embedding_inv.detach().numpy()))

    def evaluate_precritiquing(self):
        hitatone=0
        hitatthree=0
        hitatfive=0
        hitatten=0
        hitattwenty=0
        counter=0
        user_counter=0
        user_posterior=torch.ones(self.emb_dim).to(self.device)
        for user_id in tqdm(self.users_list):
            ground_truth=0
            recommender= Recommender(self.dataset,self.model,user_id,ground_truth,"pre",user_posterior,self.items_embeddings_head,self.items_embeddings_tail,self.users_embeddings_head_proj[user_id-self.users_list[0]],self.users_embeddings_tail_proj[user_id-self.users_list[0]])
            ranked_indices = recommender.pre_critiquing_new()

            for ground_truth in self.users_likes[user_id]:
                rank = int(np.where(ranked_indices==self.items_index[ground_truth])[0])
                if rank<2:
                    hitatone +=1
                if rank<4:
                    hitatthree += 1
                if rank<6:
                    hitatfive += 1
                if rank<11:
                    hitatten += 1
                if rank<21:
                    hitattwenty +=1
                counter += 1
            user_counter += 1

            wandb.log({"step":user_counter})
        hitatone_normalized= hitatone/counter
        hitatthree_normalized= hitatthree/counter
        hitatfive_normalized= hitatfive/counter
        hitatten_normalized= hitatten/counter
        hitattwenty_normalized= hitattwenty/counter
        wandb.log({"hit@1":hitatone_normalized,"hit@3":hitatthree_normalized,"hit@5":hitatfive_normalized,"hit@10":hitatten_normalized,"hit@20":hitattwenty_normalized})
        return hitatone_normalized, hitatthree_normalized, hitatfive_normalized, hitatten_normalized, hitattwenty_normalized
