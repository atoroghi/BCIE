import torch
from dataset import Dataset
from SimplE import SimplE
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import numpy as np
import random
import operator


class Recommender:
    def __init__(self, loaddataset, model, user_id, ground_truth,pre_or_post, user_posterior,items_embeddings_head,items_embeddings_tail,user_embeddings_head_proj,user_embeddings_tail_proj):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.model = torch.load(model_path, map_location=self.device)
        self.model = model
        self.model.eval()
        self.dataset = loaddataset
        self.user_id = user_id
        self.items_list = self.dataset.items
        self.items_index = dict(zip(self.items_list,list(range(0,len(self.items_list)))))
        self.items_index_inverse=dict(zip(list(range(0,len(self.items_list))),self.items_list))
        self.ground_truth = ground_truth
        self.pre_or_post = pre_or_post
        self.user_posterior = user_posterior
        self.items_embeddings_head = items_embeddings_head
        self.items_embeddings_tail = items_embeddings_tail
        self.user_embedding_head_proj = user_embeddings_head_proj
        self.user_embedding_tail_proj = user_embeddings_tail_proj


    def pre_critiquing_new(self):
      m1 = np.multiply(self.user_embedding_head_proj ,self.items_embeddings_tail)
      m2 = np.multiply(self.user_embedding_tail_proj ,self.items_embeddings_head)
      scores = np.clip(np.sum(m1 + m2, axis=1),-40,40)
      ranked_items_indices = scores.argsort()[::-1]
      return ranked_items_indices


    def pre_critiquing_recommendation(self):

        scores= np.clip(np.sum((np.multiply(self.user_embedding_head_proj ,self.items_embeddings_tail)+np.multiply(self.user_embedding_tail_proj,self.items_embeddings_head)),axis=1),-40,40)
        ranked_items_indices=scores.argsort()[::-1]
        rank = int(np.where(ranked_items_indices==self.items_index[self.ground_truth])[0])
        recommended_items = [self.items_index_inverse[k] for k in ranked_items_indices [0:20]]
       #greaterlist = [item for item, scores in scores_dict.items() if scores >= scores_dict[self.ground_truth]]
        return recommended_items, rank

    # the latter one is ground_truth rank before critiquing

    def post_critiquing_recommendation(self, user_posterior,gt):
        self.ground_truth = gt
        scores= np.clip(np.sum((np.multiply(user_posterior,self.items_embeddings_tail)+np.multiply(self.user_embedding_tail_proj,self.items_embeddings_head)),axis=1),-40,40)
        ranked_items_indices=scores.argsort()[::-1]
        rank = int(np.where(ranked_items_indices==self.items_index[self.ground_truth])[0])
        recommended_items = [self.items_index_inverse[k] for k in ranked_items_indices [0:20]]
        return recommended_items, rank


    ### inputs to the "select_Critique" method are: facts about the ground truth and facts about the recommended items as well as the critique mode and the dictionary containing popularities of each object
    ### Also, we input the facts in which the gt is placed in their tail to differentiate between objects and relations
    def select_critique(self,critique_selection_data,rec_facts,critique_mode,pop_counts,items_facts_tail_gt):
      facts_diff={}
      #we want to count how many times each ground truth facts is satisfied by the facts about recommended items
      for fact in critique_selection_data:
            condition = (fact[0] == rec_facts[:, 0]) & (fact[1] == rec_facts[:, 1])
            facts_diff[tuple(fact)]=np.count_nonzero(condition)

      if critique_mode=="random":
        # in this case, we want to randomly choose a critique from either head or tail
        if critique_selection_data.size:
          #we're only keeping the facts that are not satisfied by all recommended items
            candidate_facts=[k for k, v in facts_diff.items() if v < 20]
            critique_fact=random.choice(candidate_facts)
            if any((items_facts_tail_gt[:]==np.array((critique_fact)+(-1,))).all(1)):
              object=critique_fact[0]
            else:
              object=critique_fact[1]
        else:
          object , critique_fact = None, None
        # In this case, we want to select the fact about the most famous object as the crtique
      if critique_mode=="pop":
          if critique_selection_data.size:
            candidate_facts=[k for k, v in facts_diff.items() if v < 20]
            popularities={}
            facts={}
            for fact in candidate_facts:
              # Checking if the object is in the head (gt in the tail)
              if any((items_facts_tail_gt[:]==np.array((fact)+(-1,))).all(1)):
                candidate_object=fact[0]
                popularities[candidate_object]=pop_counts[fact[0]]
                facts[candidate_object]=fact
              else:
                candidate_object=fact[1]
                popularities[candidate_object]=pop_counts[fact[1]]
                facts[candidate_object]=fact
            object=max(popularities.items(), key=operator.itemgetter(1))[0]
            critique_fact=facts[object]
          else:
            object , critique_fact = None, None

        # in this case, we should take the fact that deviates the most from the facts related to recommended items, i.e. least satisfaction counts
      if critique_mode=="diff":        
          if bool(facts_diff):
            # this is the number of times the most deviating fact is satisfied
            minval=min(facts_diff.values())
          # in the most deviating fact, gt is in the tail 
            # we might have multiple facts satisfied minval times. In this case, we choose the most famous object as the critique
            candidates_list = [k for k, v in facts_diff.items() if v == minval]
            if len(candidates_list)>1:
              popularities={}
              facts={}
              for fact in candidates_list:
                if any((items_facts_tail_gt[:]==np.array((fact)+(-1,))).all(1)):
                  candidate_object=fact[0]
                  popularities[candidate_object]=pop_counts[fact[0]]
                  facts[candidate_object]=fact
                else:
                  candidate_object=fact[1]
                  popularities[candidate_object]=pop_counts[fact[1]]
                  facts[candidate_object]=fact
              object=max(popularities.items(), key=operator.itemgetter(1))[0]
              critique_fact=facts[object]
            else:
              if any((items_facts_tail_gt[:]==np.array((candidates_list[0])+(-1,))).all(1)):
                object=candidates_list[0][0]
              
              else:
                object=candidates_list[0][1]
              critique_fact=candidates_list[0]

          else:
            object , critique_fact = None, None
      return object , critique_fact

    def remove_chosen_critiques(self, critiquing_candidate, previous_critiques):
        for end in ["head", "tail"]:
            for record in previous_critiques[end]:
                if len(critiquing_candidate[end]) > 1:
                    critiquing_candidate[end].remove(record)

        return critiquing_candidate

    def obj2item(self,obj,data):
      objkg=data[np.where((data[:, 0] == obj) | (data[:, 2] == obj))]
      objkg=np.delete(objkg,1,1)
      mapped_items=np.intersect1d(self.items,data)
      return mapped_items
