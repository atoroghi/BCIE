import torch
from dataset import Dataset
import numpy as np
import os, sys, time
from os import listdir
from os.path import isfile, join
from dataload import LoadDataset
from tqdm import tqdm
from recommender import Recommender
import wandb

class Tester:
    def __init__(self, model_path, valid_or_test, loaddataset, emb_dim, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.dataset = args.dataset
        self.valid_or_test = valid_or_test
        self.emb_dim = emb_dim
        self.model_path = model_path
        self.loaddataset = loaddataset
        self.items_list = loaddataset.rec_items
        self.users_likes = loaddataset.user_likes_map
        
        self.users_list = list(loaddataset.users)
        self.items_embeddings_head = np.zeros([len(self.items_list),self.emb_dim])
        self.items_embeddings_tail = np.zeros([len(self.items_list),self.emb_dim])
        
        # builds dicts to go from item id to array index
        self.items_index = dict(zip(self.items_list,list(range(0,len(self.items_list)))))
        self.items_index_inverse = dict(zip(list(range(0,len(self.items_list))),self.items_list))
        
        self.users_embeddings_head_proj = np.zeros([len(self.users_list), self.emb_dim])
        self.users_embeddings_tail_proj = np.zeros([len(self.users_list), self.emb_dim])
        
        if args.user == 'armin':
            wandb.login(key="d606ae06659873794e6a1a5fb4f55ffc72aac5a1")
            wandb.init(project="pre-critiquing",config={"lr": 0.1},reinit=True,settings=wandb.Settings(start_method="fork"))
            os.environ['WANDB_API_KEY']='d606ae06659873794e6a1a5fb4f55ffc72aac5a1'
            os.environ['WANDB_USERNAME']='atoroghi'
            wandb.config.update(args, allow_val_change=True)

        # get user and item embeddings
        for item in self.items_list:
            t = torch.tensor([item]).long()
            item_embedding_head = self.model.ent_h_embs(t)
            item_embedding_tail = self.model.ent_t_embs(t)
            to_head = (item_embedding_head.detach().numpy()).reshape((1,self.emb_dim))
            to_tail = (item_embedding_tail.detach().numpy()).reshape((1,self.emb_dim))
            
            index = self.items_index[item]
            self.items_embeddings_head[index] = to_head
            self.items_embeddings_tail[index] = to_tail

        # UPDATE: moved these lines up to compute a single time
        r = torch.tensor([47]).long()
        likes_embedding = self.model.rel_embs(r)
        likes_embedding_inv = self.model.rel_inv_embs(r)
        for user in self.users_list:
            h = torch.tensor([user]).long()
            users_embedding_head = self.model.ent_h_embs(h)
            users_embedding_tail = self.model.ent_t_embs(h)
            self.users_embeddings_head_proj[user-self.users_list[0]]=np.multiply((users_embedding_head.detach().numpy()),(likes_embedding.detach().numpy()))
            self.users_embeddings_tail_proj[user-self.users_list[0]]=np.multiply((users_embedding_tail.detach().numpy()),(likes_embedding_inv.detach().numpy()))

    # TODO: this whole thing can be run on gpu faster
    # (at least most of it)
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
            # TODO: is this used?
            ground_truth = 0
            
            # TODO: do we need to remake this each time?
            # pass whole thing or just update embedding proj
            recommender = Recommender(
                self.loaddataset,
                self.model,
                user_id,
                ground_truth,
                "pre",
                user_posterior,
                self.items_embeddings_head,
                self.items_embeddings_tail,
                self.users_embeddings_head_proj[user_id-self.users_list[0]],
                self.users_embeddings_tail_proj[user_id-self.users_list[0]]
            )

            # TODO: all users x all items?
            # get scores
            ranked_indices = recommender.pre_critiquing_new()

            # compute hits
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
                    hitattwenty += 1
                counter += 1
            user_counter += 1

            if self.args.user == 'armin':
                wandb.log({"step":user_counter})
        
        hitatone_normalized = hitatone/counter
        hitatthree_normalized = hitatthree/counter
        hitatfive_normalized = hitatfive/counter
        hitatten_normalized = hitatten/counter
        hitattwenty_normalized = hitattwenty/counter
        wandb.log({"hit@1":hitatone_normalized,"hit@3":hitatthree_normalized,"hit@5":hitatfive_normalized,"hit@10":hitatten_normalized,"hit@20":hitattwenty_normalized})
        return hitatone_normalized, hitatthree_normalized, hitatfive_normalized, hitatten_normalized, hitattwenty_normalized