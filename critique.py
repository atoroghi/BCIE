from cmath import tau
import os, pickle, sys, argparse, yaml
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from updater import Updater, beta_update, beta_update_indirect
from updater_indirect import Updater_Indirect
from dataload import DataLoader
from tester import get_array, get_emb, get_scores, get_rank, GetGT
from recommender import select_critique, beta_crit, remove_chosen_critiques, obj2item
from utils.plots import RankTrack, rank_plot, save_metrics_critiquing
from utils.updateinfo import UpdateInfo
from launch import get_model_args
from argparse import Namespace
import time

def get_args_critique():
    # TODO: load all hp's from load_name
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help='name of folder where results are saved')
    parser.add_argument('-load_name', default='dev', type=str, help='name of folder where model is')
    parser.add_argument('-fold', default=0, type=int, help='fold number')

    # TODO: list for etta?
    parser.add_argument('-user_prec', default=1e5, type=float, help='prior cov')
    parser.add_argument('-default_prec', default=1e-2, type=float, help='likelihood precision')
    parser.add_argument('-z_prec', default=1e-2, type=float, help='item distribution precision indirect case')
    parser.add_argument('-etta_0', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-etta_1', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-etta_2', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-etta_3', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-etta_4', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-num_users', default=100, type=int, help='number of users')

    # TODO: put in asserts
    # single vs mult
    parser.add_argument('-critique_target', default='item', type=str, help='object or item')

    # single only
    parser.add_argument('-evidence_type', default='indirect', type=str, help='direct or indirect')
    
    # likelihood
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gauss')
    parser.add_argument('-critique_mode', default='random', type=str, help='random or pop or diff')

    args = parser.parse_args()
    return args

# stack facts to be [-1, rel, node] or [head, rel, -1]
def fact_stack(head, tail):
    if head.shape[0] != 0 and tail.shape[0] != 0:
        head = np.hstack((-np.ones((head.shape[0], 1)), head))
        tail = np.hstack((tail, -np.ones((tail.shape[0], 1))))
        return np.vstack((head, tail)).astype(np.int32)
    if head.shape[0] != 0:
        return np.hstack((-np.ones((head.shape[0], 1)), head)).astype(np.int32)
    if head.shape[0] != 0:
        return np.hstack((tail, -np.ones((tail.shape[0], 1)))).astype(np.int32)

# to numpy array
def unpack_dic(dic, ids):
    array = None
    for i in ids:
        out = np.array(dic[i])
        if array is None: array = out
        else: array = np.vstack((array, out))
    return array

def stack_emb(get_emb, model, items): # currying would be useful here
    # TODO: this should be random....
    for i, x in enumerate(items[:10]):
        out = get_emb(torch.tensor(x), model)
        if i == 0:
            array_f = out[0]
            array_inv = out[0]
        else:
            array_f = torch.stack(array_f)
            array_inv = torch.stack(array_inv)

# get dics w info about all triplets
def get_dics(args):
    data_path = 'datasets/' + args.dataset
    # all triples when item is head
    with open(os.path.join(data_path, 'items_facts_head.pkl'), 'rb') as f:
        item_facts_head = pickle.load(f)
    # all triples when item is tail
    with open(os.path.join(data_path, 'items_facts_tail.pkl'), 'rb') as f:
        item_facts_tail = pickle.load(f)
    # all items tied to a relationship
    with open(os.path.join(data_path, 'obj2items.pkl'), 'rb') as f:
        obj2items = pickle.load(f)
    # total count for each node
    with open(os.path.join(data_path, 'pop_counts.pkl'), 'rb') as f:
        pop_counts = pickle.load(f)
    return (item_facts_head, item_facts_tail, obj2items, pop_counts)

# return info about each of the priors
class Priors:
    #TODO: this is for more complex prior
    def __init__(self, crit_args, model_args):
        # we assume this (this is a hp)
        self.user_prec_f = crit_args.user_prec * np.eye(model_args.emb_dim)
        self.user_prec_inv = crit_args.user_prec * np.eye(model_args.emb_dim)

        # the model defines this. N(0, lambda*I)
        # prior over items for I^2
        #if crit_args.evidence_type == 'indirect':

        # Armin: shouldn't this be precision? so 1/reg_lambda ? 
        self.z_prec_f = model_args.reg_lambda * np.eye(model_args.emb_dim)
        self.z_prec_inv = model_args.reg_lambda * np.eye(model_args.emb_dim)
        self.z_mean_f = np.zeros(model_args.emb_dim)
        self.z_mean_inv = np.zeros(model_args.emb_dim)

# get d embedding, used for p(u | d) baysian update
def get_d(model, crit, rel_emb, obj2items, crit_args, model_args):
    (crit_node, crit_rel) = crit
    
    # single embedding of crit node
    if crit_args.critique_target == 'object':
        node_emb = get_emb(crit_node, model)
        d_f = rel_emb[0,0] * node_emb[1]
        d_inv = rel_emb[0,1] * node_emb[0]

    # TODO: clean this up.. too many lines!
    # make stack of likes * items related to feedback node
    elif crit_args.critique_target == 'item':
        liked_items_list = obj2items[crit_node]

        # get and stack things
        liked_embeddings_list_f = []
        liked_embeddings_list_inv = []
        # TODO: this should be random or something...
        # Armin: this is not changed yet right? shall I change it?
        for x in liked_items_list[:10]:
            liked_embeddings_list_f.append(get_emb(torch.tensor(x),model)[1])
            liked_embeddings_list_inv.append(get_emb(torch.tensor(x),model)[0])
        liked_embeddings_f = torch.stack(liked_embeddings_list_f, dim=0)
        liked_embeddings_inv = torch.stack(liked_embeddings_list_inv, dim=0)
        true_object_embedding_f = torch.reshape(liked_embeddings_f,(liked_embeddings_f.shape[0], model_args.emb_dim))
        true_object_embedding_inv = torch.reshape(liked_embeddings_inv,(liked_embeddings_inv.shape[0], model_args.emb_dim))
    
        d_f = rel_emb[0,0] * true_object_embedding_f
        d_inv = rel_emb[0,1] * true_object_embedding_inv
    return (d_f, d_inv)

# main loop
def critiquing(crit_args, mode):
# setup
#################################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args = Namespace()
    etta = [crit_args.etta_0, crit_args.etta_1, crit_args.etta_2, crit_args.etta_3, crit_args.etta_4]

    # load model and get parameter from file
    save_path = os.path.join('results', crit_args.test_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(crit_args.load_name, 'info.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        for key in yml.keys():
            if key != "test_name":
                setattr(model_args, key, yml[key])
    # TODO: save these in yaml file
    model_args.learning_rel = 'learn'
    model_args.type_checking = 'yes'
    save_dict = {}
    for k, v in crit_args.items():
        save_dict.update({k : str(v)})

     with open(os.path.join(save_path, 'crit hps.yml'), 'w') as f:
                yaml.dump(save_dict, f, sort_keys=False,
                        default_flow_style=False)

    # load model
    model_path = os.path.join(crit_args.load_name, 'models/best_model.pt')
    model = torch.load(model_path).to(device)

    # load dataset + dictionaries
    dataloader = DataLoader(model_args)
    (item_facts_head, item_facts_tail, obj2items, pop_counts) = get_dics(model_args)


    # make arrays with embeddings, and dict to map 
    rec_h, rec_t, id2index, index2id = get_array(model, dataloader, model_args, rec=True)
    item_emb = (rec_h, rec_t)

    # get all relationships
    with torch.no_grad():
        rels = dataloader.num_rel
        r = torch.linspace(0, rels - 1, rels).long().to(device)
        rel_f = model.rel_embs(r)
        rel_inv = model.rel_inv_embs(r)
        rel_emb = torch.stack((rel_f, rel_inv))
        rel_emb = torch.permute(rel_emb, (1,0,2))
    likes_rel = rel_emb[0]

    # all users, gt + data (based on test / train)
    get_gt = GetGT(dataloader.fold, mode)
    data = dataloader.rec_test if mode == 'test' else dataloader.rec_val
    all_users = torch.tensor(data[:, 0]).cpu().numpy()

    # main test loop (each user)
##############################################################
##############################################################
    rank_track = None
    for i, user in enumerate(all_users):
        print(i)
        #if i == 10: break

        # get ids of top k recs, and all gt from user
        user_emb = get_emb(user, model)
        ranked = get_scores(user_emb, rel_emb[0], item_emb, dataloader, model_args.learning_rel)
        rec_ids = [index2id[int(x)] for x in ranked[:20]]
        test_gt, all_gt, train_gt = get_gt.get(user)

        # iterature through all gt for single user
        for j, gt in enumerate(test_gt):
            # stack facts, either [-1, rel, tail] or [head, rel, -1]
            ht_facts = fact_stack(item_facts_head[gt], item_facts_tail[gt])
            if ht_facts.shape[0] < crit_args.session_length:
                continue

            # save initial rank and previous user crits 
            sub_track = np.empty(crit_args.session_length + 1)
            sub_track[0] = get_rank(ranked, [gt], all_gt, id2index) 

            # a few sessions for each user 
            for sn in range(crit_args.session_length):
            ##############################################################
            ##############################################################
                if sn == 0: 

                    update_info = UpdateInfo(user_emb, etta, crit_args, model_args, device, likes_emb=likes_rel)

                # TODO: move this somewhere else, not important...
                # get all facts related to top n movies rec (from model)
                if crit_args.critique_target == 'item':
                    rec_facts_head = unpack_dic(item_facts_head, rec_ids)
                    rec_facts_tail = unpack_dic(item_facts_tail, rec_ids)
                    rec_facts = np.vstack([rec_facts_head, rec_facts_tail])

                # select a crit (user action) and remove it from pool
                #crit_node, crit_pair = select_critique(ht_facts, rec_facts, crit_args.critique_mode, pop_counts, items_facts_tail_gt)


                crit, ht_facts = beta_crit(ht_facts) # crit in (node, rel) format

                crit_rel_emb = rel_emb[crit[1]] 

                # get d for p(user | d) bayesian update
                d = get_d(model, crit, rel_emb, obj2items, crit_args, model_args)
                update_info.store(d=d, crit_rel_emb=crit_rel_emb)
                if crit_args.evidence_type == 'direct':
                    beta_update(update_info, sn, crit_args, model_args, device)
                if crit_args.evidence_type == 'indirect':
                    beta_update_indirect(update_info, sn, crit_args, model_args, device)

                # fast updater


                # track rank in training
                new_user_emb, _ = update_info.get_priorinfo()
                ranked = get_scores(new_user_emb, rel_emb[0], item_emb, dataloader, model_args.learning_rel)
                rank = get_rank(ranked, [gt], all_gt, id2index)
                sub_track[sn + 1] = rank

            # update w new data
            st = np.expand_dims(sub_track, axis=0).astype(np.int32)
            if rank_track is None: rank_track = st
            else: rank_track = np.concatenate((rank_track, st))
        print("no of processed users: {}".format(i))


    # save results
    if mode == 'val':
        mrr = save_metrics_critiquing(rank_track, crit_args.test_name, mode)
        return mrr
    else:
        print('not done')
        sys.exit()
        save_metrics_critiquing(rank_track, args.test_name, mode)

# TODO: just make this a function?? 
if __name__ == '__main__':
    crit_args = get_args_critique()
    critiquing(crit_args, 'val')

# TODO: make fast updaters...
#if crit_args.evidence_type == 'direct':
    #updater_f = Updater(d_f, y, user_emb_f, prior.z_prec_f, crit_args, model_args, device, etta)
    #updater_inv = Updater(d_inv, y, user_emb_inv, prior.z_prec_inv, crit_args, model_args, device, etta)

    #user_emb_f, user_prior_f = updater_f.compute_laplace_approximation()
    #user_emb_inv, user_prior_inv = updater_inv.compute_laplace_approximation()

#elif crit_args.evidence_type == 'indirect':
    #updater_f = Updater_Indirect(d_f, y, user_emb_f, prior.user_prec_f, prior.z_pre_f, model_args, device, etta, likes_rel)
    #updater_inv = Updater_Indirect(d_inv, y, user_emb_inv, prior.user_prec_inv, prior.z_pre_inv, model_args, device, etta, likes_rel)
    #user_emb_f, tau_prior_f = updater_f.compute_laplace_approximation()
    #user_emb_inv, tau_prior_inv = updater_inv.compute_laplace_approximation()
