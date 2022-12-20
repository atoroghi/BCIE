import torch, os, sys, random, operator
from itertools import count
import numpy as np

# either head or tail isn't -1 [h, r, t]
def get_node(t):
    if t[0] == -1: return t[2]
    else: return t[0]

# [node, rel]
def return_crit(t):
    if t[0] == -1: return (t[2], t[1])
    else: return (t[0], t[1])

# normalize list of vectors:
def list_norm(x):
    norm = torch.linalg.norm(x, axis=1)
    return (x.T / norm).T

def norm(x): return x / torch.linalg.norm(x)

# get crit of item that is most similar in emb space
def test_crit(gt, model, item_emb, index2id, device):
    a = torch.tensor(gt).to(device)
    gt_emb = (model.ent_h_embs(a), model.ent_t_embs(a))
    gt_emb = (norm(gt_emb[0]), norm(gt_emb[1]))
    item_emb = (list_norm(item_emb[0]), list_norm(item_emb[1]))
    
    for_prod = torch.sum(gt_emb[0] * item_emb[1], axis=1)
    inv_prod = torch.sum(gt_emb[1] * item_emb[0], axis=1)
    scores = (for_prod + inv_prod) / 2
    (scores, ind) = torch.sort(scores, descending=True)

    crit = (index2id[ind[0].cpu().item()], 0)
    return crit, scores[0].cpu().item()

# actual critique selection function
# ht facts is facts of gt, other are facts about top k rec (naming is terrible)
# pop count is number of connections for each node in kg
def crit_selector(gt_facts, rec_facts, critique_mode, pop_counts):
    # TODO: there must be a better way...
    # get counts for each triplet
    count = np.zeros(gt_facts.shape[0])
    for i in range(gt_facts.shape[0]):
        for j in range(len(rec_facts)):
            if (gt_facts[i] == rec_facts[j]).all(axis=1).any():
                count[i] += 1

    # inds that are unique (aren't in all rec_facts)        
    inds = np.where(count < len(rec_facts))[0]
    #inds = np.random.permutation(inds) # TODO: this is extra silly

    # select random triplet
    if critique_mode == 'random': 
        ind = np.random.choice(inds)
        return return_crit(gt_facts[ind]), gt_facts[ind]

    # select tripet that is most popular
    if critique_mode == 'pop':
        best_ind = inds[0]
        most_pop = pop_counts[get_node(gt_facts[0])]

        for i, ind in enumerate(inds):
            if pop_counts[get_node(gt_facts[i])] > most_pop: best_ind = inds[i]
        return return_crit(gt_facts[best_ind]), gt_facts[np.argmin(count)]
    
    # select tripet that is least popular
    if critique_mode == 'antipop':
        best_ind = inds[0]
        most_pop = pop_counts[get_node(gt_facts[0])]

        for i, ind in enumerate(inds):
            if pop_counts[get_node(gt_facts[i])] < most_pop: best_ind = inds[i]
        return return_crit(gt_facts[best_ind]), gt_facts[np.argmin(count)]

    # get triplet that has least similarity
    if critique_mode == 'diff': return return_crit(gt_facts[np.argmin(count)]), gt_facts[np.argmin(count)]
    