import torch
from dataset import Dataset
import numpy as np
from measure import Measure
import os
from os import listdir
from os.path import isfile, join
from dataload import LoadDataset
from tqdm import tqdm
from recommender import Recommender
import wandb


class Tester:
    def __init__(self, dataset, model_path, valid_or_test,loaddataset,emb_dim,args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location = self.device)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        #self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())
        self.emb_dim=emb_dim
        self.model_path=model_path
        self.loaddataset=loaddataset
        self.items_list=loaddataset.rec_items
        self.users_likes= loaddataset.user_likes_map
        #self.users_list=self.users_likes.keys()
        self.users_list=list(self.users_likes.keys())
        self.items_embeddings_head=(-100)*np.ones([np.max(self.items_list)+1,self.emb_dim])
        self.items_embeddings_tail=(-100)*np.ones([np.max(self.items_list)+1,self.emb_dim])
        self.users_embeddings_head_proj=np.zeros([np.max(self.users_list)-self.users_list[0]+1,self.emb_dim])
        self.users_embeddings_tail_proj=np.zeros([np.max(self.users_list)-self.users_list[0]+1,self.emb_dim])
        wandb.login(key="d606ae06659873794e6a1a5fb4f55ffc72aac5a1")
        wandb.init(project="pre-critiquing",config={"lr": 0.1})
        os.environ['WANDB_API_KEY']='d606ae06659873794e6a1a5fb4f55ffc72aac5a1'
        os.environ['WANDB_USERNAME']='atoroghi'
        wandb.config.update(args,allow_val_change=True)

        for item in self.items_list:
            h = torch.tensor([0]).long().to(self.device)
            r = torch.tensor([0]).long().to(self.device)
            t = torch.tensor([item]).long().to(self.device)
            _, _, _, item_embedding_tail, _, _, item_embedding_head = self.model(h, r, t)

            to_head=(item_embedding_head.detach().numpy()).reshape((1,self.emb_dim))
            to_tail=(item_embedding_tail.detach().numpy()).reshape((1,self.emb_dim))
            
            self.items_embeddings_head[item]=to_head
            self.items_embeddings_tail[item]=to_tail
        for user in self.users_list:
            h = torch.tensor([user]).long().to(self.device)
            r = torch.tensor([47]).long().to(self.device)
            t = torch.tensor([0]).long().to(self.device)
            _, users_embedding_head, likes_embedding, _, users_embedding_tail, likes_embedding_inv, _ = self.model(h, r, t)
            self.users_embeddings_head_proj[user-self.users_list[0]]=np.multiply((users_embedding_head.detach().numpy()),(likes_embedding.detach().numpy()))
            self.users_embeddings_tail_proj[user-self.users_list[0]]=np.multiply((users_embedding_tail.detach().numpy()),(likes_embedding_inv.detach().numpy()))

    def evaluate_precritiquing(self):
        hitatone=0
        hitatthree=0
        hitatfive=0
        hitatten=0
        hitattwenty=0
        counter=0
        user_posterior=torch.ones(self.emb_dim).to(self.device)
        for user_id in tqdm(self.users_list):
            for ground_truth in self.users_likes[user_id]:
                recommender= Recommender(self.loaddataset,self.model,user_id,ground_truth,"pre",user_posterior,self.items_embeddings_head,self.items_embeddings_tail,self.users_embeddings_head_proj[user_id-self.users_list[0]],self.users_embeddings_tail_proj[user_id-self.users_list[0]])
                _, rank= recommender.pre_critiquing_recommendation()
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
            print(hitatone/counter)
            print(hitatthree/counter)
            print(hitatfive/counter)
            print(hitatten/counter)
            print(hitattwenty/counter)
        hitatone_normalized= hitatone/counter
        hitatthree_normalized= hitatthree/counter
        hitatfive_normalized= hitatfive/counter
        hitatten_normalized= hitatten/counter
        hitattwenty_normalized= hitattwenty/counter
        wandb.log({"hit@1":hitatone_normalized,"hit@3":hitatthree_normalized,"hit@5":hitatfive_normalized,"hit@10":hitatten_normalized,"hit@20":hitattwenty_normalized})
        return hitatone_normalized, hitatthree_normalized, hitatfive_normalized, hitatten_normalized, hitattwenty_normalized


    #def get_rank(self, sim_scores):#assuming the test fact is the first one
    #    return (sim_scores >= sim_scores[0]).sum()

    #def create_queries(self, fact, head_or_tail):
    #    head, rel, tail = fact
    #    if head_or_tail == "head":
        #    return [(i, rel, tail) for i in range(self.dataset.num_ent())]
      #  elif head_or_tail == "tail":
      #      return [(head, rel, i) for i in range(self.dataset.num_ent())] """

   # """def add_fact_and_shred(self, fact, queries, raw_or_fil):
   #     if raw_or_fil == "raw":
    #        result = [tuple(fact)] + queries
     #   elif raw_or_fil == "fil":
      #      result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)

       # return self.shred_facts(result)

    # def replace_and_shred(self, fact, raw_or_fil, head_or_tail):
    #     ret_facts = []
    #     head, rel, tail = fact
    #     for i in range(self.dataset.num_ent()):
    #         if head_or_tail == "head" and i != head:
    #             ret_facts.append((i, rel, tail))
    #         if head_or_tail == "tail" and i != tail:
    #             ret_facts.append((head, rel, i))

    #     if raw_or_fil == "raw":
    #         ret_facts = [tuple(fact)] + ret_facts
    #     elif raw_or_fil == "fil":
    #         ret_facts = [tuple(fact)] + list(set(ret_facts) - self.all_facts_as_set_of_tuples)

    #     return self.shred_facts(ret_facts)
    
    #def test(self):
     #   settings = ["raw", "fil"] if self.valid_or_test == "test" else ["fil"]
        
      #  for i, fact in enumerate(self.dataset.data[self.valid_or_test]):
       #     for head_or_tail in ["head", "tail"]:
        #        queries = self.create_queries(fact, head_or_tail)
         #       for raw_or_fil in settings:
          #          h, r, t = self.add_fact_and_shred(fact, queries, raw_or_fil)
           #         sim_scores,_,_,_ = self.model(h, r, t)
            #        rank = self.get_rank(sim_scores)
             #       self.measure.update(rank, raw_or_fil)

        #self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        #self.measure.print_()
        #return self.measure.mrr["fil"]

    #def shred_facts(self, triples):
     #   heads  = [triples[i][0] for i in range(len(triples))]
      #  rels   = [triples[i][1] for i in range(len(triples))]
       # tails  = [triples[i][2] for i in range(len(triples))]
        #return torch.LongTensor(heads).to(self.device), torch.LongTensor(rels).to(self.device), torch.LongTensor(tails).to(self.device)

    #def allFactsAsTuples(self):
     #   tuples = []
      #  for spl in self.dataset.data:
       #     for fact in self.dataset.data[spl]:
        #        tuples.append(tuple(fact))
        #
        #return tuples"""



    