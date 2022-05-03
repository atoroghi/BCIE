import torch
from dataset import Dataset
from SimplE import SimplE
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
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
        self.items_list = self.dataset.rec_items
        self.items_index=dict(zip(self.items_list,list(range(0,len(self.items_list)))))
        self.items_index_inverse=dict(zip(list(range(0,len(self.items_list))),self.items_list))
        self.ground_truth = ground_truth
        self.pre_or_post = pre_or_post
        self.user_posterior = user_posterior
        self.items_embeddings_head = items_embeddings_head
        self.items_embeddings_tail = items_embeddings_tail
        self.user_embedding_head_proj = user_embeddings_head_proj
        self.user_embedding_tail_proj = user_embeddings_tail_proj


    # self.user_id=self.dataset.ent2id[user_no]
    # self.items_list=self.dataset.eligible_items[self.user_id]

    def pre_critiquing_new(self):
      scores=np.clip(np.sum(np.multiply(self.user_embedding_head_proj ,self.items_embeddings_tail)+np.multiply(self.user_embedding_tail_proj ,self.items_embeddings_head),axis=1),-40,40)
      ranked_items_indices=scores.argsort()[::-1]
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


 #   def select_critique(self,data,critique_mode,recommended_items):
 #       global critique
 #       gt_facts_head = data[np.where((data[:, 0] == self.ground_truth))]
 #       gt_facts_tail = data[np.where((data[:, 2] == self.ground_truth))]
 #       rec_facts_head = np.array([])
 #       rec_facts_tail = np.array([])
 #       for item in recommended_items:
 #           rec_facts_head = np.vstack(
 #               [rec_facts_head, data[
 #                   (np.where((data[:, 0] == item) ))]]) if rec_facts_head.size \
 #               else data[(np.where((data[:, 0] == item)))]
 #           rec_facts_tail = np.vstack(
 #               [rec_facts_tail, data[
 #                   (np.where((data[:, 2] == item)))]]) if rec_facts_tail.size \
 #               else data[(np.where((data[:, 2] == item)))]
 #       facts_diff_head={}
 #       facts_diff_tail={}
 #       for fact in gt_facts_head:
 #           condition = (fact[1] == rec_facts_head[:, 1]) & (fact[2] == rec_facts_head[:, 2])
 #           facts_diff_head[tuple(fact[1:])]=np.count_nonzero(condition)
 #       for fact in gt_facts_tail:
 #           condition= (fact[1] == rec_facts_tail[:, 1]) & (fact[0] == rec_facts_tail[:, 0])
 #           facts_diff_tail[tuple(fact[:2])] = np.count_nonzero(condition)
 #       if critique_mode=="random":
 #           critique_candidates={}
 #           if bool(facts_diff_tail):
 #             candidate_facts_tail=dict((k, v) for k, v in facts_diff_tail.items() if v <len(recommended_items))
 #             tail_candidate=(0,0)
 #             if list(candidate_facts_tail):
 #               tail_candidate=random.choice(list(candidate_facts_tail))
 #             critique_candidates["tail"]=tail_candidate
 #           if bool(facts_diff_head):
 #             candidate_facts_head=dict((k, v) for k, v in facts_diff_head.items() if v <len(recommended_items))
 #             head_candidate=(0,0)
 #             if list(candidate_facts_head):
 #               head_candidate=random.choice(list(candidate_facts_head))
 #             critique_candidates["head"]=head_candidate
 #           critique_facts=(0,0)
 #           while critique_facts==(0,0):
 #             selected_end=random.choice(list(critique_candidates))
 #             critique=critique_candidates[selected_end], selected_end
 #             critique_facts=critique[0]
 #       if critique_mode=="pop":
 #           critique_candidates={}
 #           most_famous_repeats_tail=0
 #           most_famous_repeats_head=0
 #           if bool(facts_diff_tail):
 #             candidate_facts_tail=dict((k, v) for k, v in facts_diff_tail.items() if v <len(recommended_items))
 #             freq_candidates_list_tail={}
 #             for candidate in candidate_facts_tail:
 #               freq_candidates_list_tail[candidate]=((data[:,0]==candidate[0]) | (data[:,2]==candidate[0])).sum()
 #               most_famous_repeats_tail = max(freq_candidates_list_tail.values())
 #               tail_candidate = [k for k, v in freq_candidates_list_tail.items() if v == most_famous_repeats_tail][0]
 #               critique_candidates["tail"]=tail_candidate
 #           if bool(facts_diff_head):
 #             candidate_facts_head=dict((k, v) for k, v in facts_diff_head.items() if v <len(recommended_items))
 #             freq_candidates_list_head={}
 #             for candidate in candidate_facts_head:
 #               freq_candidates_list_head[candidate]=((data[:,0]==candidate[1]) | (data[:,2]==candidate[1])).sum()
 #               most_famous_repeats_head = max(freq_candidates_list_head.values())
 #               head_candidate = [k for k, v in freq_candidates_list_head.items() if v == most_famous_repeats_head][0]
 #               critique_candidates["head"]=head_candidate
 #           if most_famous_repeats_tail>most_famous_repeats_head:
 #             critique=critique_candidates["tail"],"tail"
 #           else:
 #             critique=critique_candidates["head"],"head"
 #       if critique_mode=="diff":
 #         if bool(facts_diff_tail):
 #           minval_tail=min(facts_diff_tail.values())
 #         else:
 #           minval_tail=0
 #         if bool(facts_diff_head):
 #           minval_head=min(facts_diff_head.values())
 #         else:
 #           minval_head=0
 #         if minval_tail<minval_head and minval_tail>0:
 #           candidates_list = [k for k, v in facts_diff_tail.items() if v == minval_tail]
 #           if len(candidates_list)>1:
 #             freq_candidates_list={}
 #             for candidate in candidates_list:
 #               freq_candidates_list[candidate]=((data[:,0]==candidate[0]) | (data[:,2]==candidate[0])).sum()
 #               most_famous_repeats = max(freq_candidates_list.values())
 #               critique = [k for k, v in freq_candidates_list.items() if v == most_famous_repeats][0],"tail"
 #           else:
 #             critique=candidates_list[0],"tail"
 #         else:
 #           candidates_list = [k for k, v in facts_diff_head.items() if v == minval_head]
 #           if len(candidates_list)>1:
 #             freq_candidates_list = {}
 #             for candidate in candidates_list:
 #               freq_candidates_list[candidate] = ((data[:, 0] == candidate[1]) | (data[:, 2] == candidate[1])).sum()
 #               most_famous_repeats = max(freq_candidates_list.values())
 #               critique = [k for k, v in freq_candidates_list.items() if v == most_famous_repeats][0],"head"
 #           else:
 #             critique=candidates_list[0],"head"

  #      print(critique)

   #     return critique

    def remove_chosen_critiques(self, critiquing_candidate, previous_critiques):
        for end in ["head", "tail"]:
            for record in previous_critiques[end]:
                if len(critiquing_candidate[end]) > 1:
                    critiquing_candidate[end].remove(record)

        return critiquing_candidate

   # def get_direct_embeddings(self, head, rel, tail):
    #    h = torch.tensor([head]).long().to(self.device)
     #   r = torch.tensor([rel]).long().to(self.device)
      #  t = torch.tensor([tail]).long().to(self.device)
       # _, head_embedding, relation_embedding, tail_embedding, _, _, _ = self.model(h, r, t)
       # return head_embedding, relation_embedding, tail_embedding

    def obj2item(self,obj,data):
      objkg=data[np.where((data[:, 0] == obj) | (data[:, 2] == obj))]
      objkg=np.delete(objkg,1,1)
      mapped_items=np.intersect1d(self.items,data)
      return mapped_items
