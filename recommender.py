import torch
from dataset import Dataset
from SimplE import SimplE
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class Recommender:
    def __init__(self, dataset, model_path, user_id, ground_truth, pre_or_post, user_posterior):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.dataset = dataset
        self.user_id = user_id
        self.items = self.dataset.items
        self.ground_truth = ground_truth
        self.pre_or_post = pre_or_post
        self.user_posterior = user_posterior

    # self.user_id=self.dataset.ent2id[user_no]
    # self.items_list=self.dataset.eligible_items[self.user_id]

    def pre_critiquing_recommendation(self):
        h = torch.tensor([self.user_id]).long().to(self.device)
        r = torch.tensor([0]).long().to(self.device)
        scores_dict = {}
        for item in self.items:
            t = torch.tensor([item]).long().to(self.device)
            score, user_embedding, relation_embedding, item_embedding, _, _, _ = self.model(h, r, t)
            scores_dict[item] = score

        recommended_items = sorted(scores_dict, key=scores_dict.get, reverse=True)[:10]
        self.recommended_items = recommended_items

        greaterlist = [item for item, scores in scores_dict.items() if scores >= scores_dict[self.ground_truth]]
        return user_embedding, relation_embedding, recommended_items, len(greaterlist)

    # the latter one is ground_truth rank before critiquing

    def post_critiquing_recommendation(self, user_posterior):
        scores_dict = {}
        h = torch.tensor([0]).long().to(self.device)
        r = torch.tensor([0]).long().to(self.device)
        for item in self.items:
            t = torch.tensor([item]).long().to(self.device)
            _, _, relation_embedding, item_embedding, _, _, _ = self.model(h, r, t)
            k = relation_embedding * item_embedding
            scores_dict[item] = torch.sum(self.user_posterior * k)

        recommended_items = sorted(scores_dict, key=scores_dict.get, reverse=True)[:10]
        self.recommended_items = recommended_items
        greaterlist = [item for item, scores in scores_dict.items() if scores >= scores_dict[self.ground_truth]]
        return self.user_posterior, relation_embedding, recommended_items, len(greaterlist)

    def select_critique(self,previous_critiques_head,previous_critiques_tail):


        data = np.concatenate((self.dataset.data["train"], self.dataset.data["test"]))
        previous_critiques=np.vstack([previous_critiques_head,previous_critiques_tail])
        dims = np.maximum(previous_critiques.max(0), data.max(0)) + 1
        data = data[~np.in1d(np.ravel_multi_index(data.T, dims), np.ravel_multi_index(previous_critiques.T, dims))]
        if self.pre_or_post == "pre":
            user_embedding, _, recommended_items, gt_rank = self.pre_critiquing_recommendation()
        else:
            user_embedding, _, recommended_items, gt_rank = self.post_critiquing_recommendation()
        gt_facts_head = data[(np.where((data[:, 0] == self.ground_truth) & (data[:, 1] != 0)))]
        gt_facts_tail = data[(np.where((data[:, 2] == self.ground_truth) & (data[:, 1] != 0)))]
        rec_facts_head = np.array([])
        rec_facts_tail = np.array([])
        for item in recommended_items:
            rec_facts_head = np.vstack(
                [rec_facts_head, data[
                    (np.where((data[:, 0] == item)  & (data[:, 1] != 0)))]]) if rec_facts_head.size \
                else data[(np.where((data[:, 0] == item)))]
            rec_facts_tail = np.vstack(
                [rec_facts_tail, data[
                    (np.where((data[:, 2] == item)  & (data[:, 1] != 0)))]]) if rec_facts_tail.size \
                else data[(np.where((data[:, 2] == item)))]
        facts_diff_head={}
        facts_diff_tail={}
        for fact in gt_facts_head:
            condition = (fact[1] == rec_facts_head[:, 1]) & (fact[2] == rec_facts_head[:, 2])
            facts_diff_head[tuple(fact[1:])]=np.count_nonzero(condition)
        for fact in gt_facts_tail:
            condition= (fact[1] == rec_facts_tail[:, 1]) & (fact[0] == rec_facts_tail[:, 0])
            facts_diff_tail[tuple(fact[:2])] = np.count_nonzero(condition)
        if bool(facts_diff_tail):
            minval_tail = min(facts_diff_tail.values())
        else:
            minval_tail=0
        if bool(facts_diff_head):
            minval_head = min(facts_diff_head.values())
        else:
            minval_head=0
        if minval_tail<minval_head and minval_tail>0:
            candidates_list = [k for k, v in facts_diff_tail.items() if v == minval_tail]
            if len(candidates_list)>1:
                freq_candidates_list={}
                for candidate in candidates_list:
                    freq_candidates_list[candidate]=((data[:,0]==candidate[0]) | (data[:,2]==candidate[0])).sum()
                    most_famous_repeats = max(freq_candidates_list.values())
                    critique = [k for k, v in freq_candidates_list.items() if v == most_famous_repeats][0],"tail"
            else:
                critique=candidates_list[0],"tail"
        else:
            candidates_list = [k for k, v in facts_diff_head.items() if v == minval_head]
            if len(candidates_list)>1:
                freq_candidates_list = {}
                for candidate in candidates_list:
                    freq_candidates_list[candidate] = (
                                (data[:, 0] == candidate[1]) | (data[:, 2] == candidate[1])).sum()
                    most_famous_repeats = max(freq_candidates_list.values())
                    critique = [k for k, v in freq_candidates_list.items() if v == most_famous_repeats][0],"head"
            else:
                critique=candidates_list[0],"head"

        return critique

#        for triplet in data:
 #           for item in recommended_items:
  #              if item != self.ground_truth:
   #                 if triplet[0] == item:
    #                    if triplet[1] > 0:
     #                       recommended_critiquing_candidate["head"].append([triplet[1], triplet[2]])
      #              if triplet[2] == item:
       #                 if triplet[1] > 0:
        #                    recommended_critiquing_candidate["tail"].append([triplet[0], triplet[1]])

         #   if triplet[0] == self.ground_truth:
          #      if triplet[1] > 0:
           #         truth_critiquing_candidate["head"].append([triplet[1], triplet[2]])

   #         if triplet[2] == self.ground_truth:
    #            if triplet[1] > 0:
     #               truth_critiquing_candidate["tail"].append([triplet[0], triplet[1]])

      #  return truth_critiquing_candidate, recommended_critiquing_candidate

    def remove_chosen_critiques(self, critiquing_candidate, previous_critiques):
        for end in ["head", "tail"]:
            for record in previous_critiques[end]:
                if len(critiquing_candidate[end]) > 1:
                    critiquing_candidate[end].remove(record)

        return critiquing_candidate

   # def get_frequent(self, head_or_tail, previous_critiques):
    #    critiquing_candidate, recommended_critiquing_candidate = self.get_critique_candidates()
        # remove previous critiques from
     #   truth_critiquing_candidate = self.remove_chosen_critiques(critiquing_candidate, previous_critiques)

     #   frequencies = {}
      #  for candidate in truth_critiquing_candidate[head_or_tail]:
       #     if head_or_tail == "head":
        #        relation = candidate[0]
         #       frequencies[relation] = 0
         #   else:
          #      relation = candidate[1]
         #       frequencies[relation] = 0
        #for candidate in truth_critiquing_candidate[head_or_tail]:
           # for item in recommended_critiquing_candidate[head_or_tail]:
           #     if head_or_tail == "head":
           #         if candidate[0] == item[0] and candidate[1] != item[1]:
          #              relation = candidate[0]
         #               frequencies[relation] = frequencies[relation] + 1
        #        else:
       #             if candidate[1] == item[1] and candidate[0] != item[0]:
      #                  relation = candidate[1]
     #                   frequencies[relation] = frequencies[relation] + 1

    #    return frequencies

   # def select_critique(self, previous_critiques):

      #  critiquing_candidate, recommended_critiquing_candidate = self.get_critique_candidates()
      #  heads_frequencies = self.get_frequent("head", previous_critiques)
      #  tails_frequencies = self.get_frequent("tail", previous_critiques)
     #   truth_critiquing_candidate = self.remove_chosen_critiques(critiquing_candidate, previous_critiques)

    #    ret = None, None, None, None

        # Check for the case either of them is empty
   #     if bool(heads_frequencies) == False:
  #          heads_frequencies[0] = 0
 #       if bool(tails_frequencies) == False:
#            tails_frequencies[0] = 0

     #   head_critique = max(heads_frequencies, key=heads_frequencies.get)
     #   tail_critique = max(tails_frequencies, key=tails_frequencies.get)
      #  if heads_frequencies[head_critique] > tails_frequencies[tail_critique]:
     #       for can in truth_critiquing_candidate["head"]:
        #           if can[0] == head_critique:
            #              rel = head_critique
                    #              tail = can[1]
                    #              false_X = []
                    #              for fact in recommended_critiquing_candidate["head"]:
                        #                   if fact[0] == rel and tail != fact[1] and fact not in false_X:
                #                       false_X.append(fact)
            #        ret = "head", rel, tail, false_X
            #    else:

    #       for can in truth_critiquing_candidate["tail"]:
        #         if can[1] == tail_critique:
                    #            rel = tail_critique
            #            head = can[0]
                    #             false_X = []
                    #             for fact in recommended_critiquing_candidate["tail"]:
                #                if fact[1] == rel and fact not in false_X:
                    #                    false_X.append(fact)
        #     ret = "tail", rel, head, false_X
        #
        # return ret

    def get_direct_embeddings(self, head, rel, tail):
        h = torch.tensor([head]).long().to(self.device)
        r = torch.tensor([rel]).long().to(self.device)
        t = torch.tensor([tail]).long().to(self.device)
        _, head_embedding, relation_embedding, tail_embedding, _, _, _ = self.model(h, r, t)
        return head_embedding, relation_embedding, tail_embedding
