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
from recommender import crit_selector, test_crit
from utils.plots import RankTrack, rank_plot, save_metrics_critiquing
from utils.updateinfo import UpdateInfo
from launch import get_model_args
from argparse import Namespace
import time

def get_args_critique():
    # TODO: load all hp's from load_name
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='gausstypereg', type=str, help='name of folder where results are saved')
    parser.add_argument('-load_name', default='results/gausstypereg/train/fold_0/train_59', type=str, help='name of folder where model is')
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
    parser.add_argument('-alpha', default=0.1, type=float, help='Learning rate for GD in Laplace Approximation')
    parser.add_argument('-multi_k', default=10, type=int, help='number of samples for multi type update')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-num_users', default=50, type=int, help='number of users')

    # TODO: put in asserts
    # single vs mult
    parser.add_argument('-critique_target', default='single', type=str, help='single or multi')

    # single only
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    
    # likelihood
    parser.add_argument('-update_type', default='laplace', type=str, help='laplace or gauss')
    parser.add_argument('-critique_mode', default='random', type=str, help='random or pop or diff')

    #laplace updating only
    parser.add_argument('-map_finder', default='gd', type= str, help='cvx or gd')

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
    if tail.shape[0] != 0:
        return np.hstack((tail, -np.ones((tail.shape[0], 1)))).astype(np.int32)

def rec_fact_stack(ids, items_facts_head, items_facts_tail):
    rec_facts = []
    for rec_id in ids:
        rec_facts.append(fact_stack(items_facts_head[rec_id], items_facts_tail[rec_id]))
    return rec_facts

# to numpy array
def unpack_dic(dic, ids):
    array = None
    for i in ids:
        out = np.array(dic[i])
        if array is None: array = out
        else: array = np.vstack((array, out))
    return array

def stack_emb(get_emb, model, items, device): # currying would be useful here
    # TODO: this should be random....
    for i, x in enumerate(items[:10]):
        out = get_emb(torch.tensor(x), model, device)
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
def get_d(model, crit, rel_emb, obj2items, crit_args, model_args, device):
    (crit_node, crit_rel) = crit

    rel_emb = rel_emb[crit_rel]

    # single embedding of crit node
    if crit_args.critique_target == 'single':

        node_emb = get_emb(crit_node, model, device)
        node_emb = list_norm(node_emb)

        # TODO: this is bad bad bad bad bad 
        if crit_args.evidence_type == 'direct':
            d_f = rel_emb[0] * node_emb[1]
            d_inv = rel_emb[1] * node_emb[0]

        else: 
            d_f = node_emb[1]
            d_inv = node_emb[0]

    # TODO: clean this up.. too many lines!
    # make stack of likes * items related to feedback node
    elif crit_args.critique_target == 'multi':
        liked_items = obj2items[crit_node]
        liked_items = np.random.permutation(liked_items)

        # get and stack things
        liked_embeddings_list_f = []
        liked_embeddings_list_inv = []

        # TODO: this should be random or something...
        for x in liked_items[:crit_args.multi_k]:
            liked_embeddings_list_f.append(get_emb(torch.tensor(x), model, device)[1])
            liked_embeddings_list_inv.append(get_emb(torch.tensor(x), model, device)[0])
        liked_embeddings_f = torch.stack(liked_embeddings_list_f, dim=0)
        liked_embeddings_inv = torch.stack(liked_embeddings_list_inv, dim=0)
        true_object_embedding_f = torch.reshape(liked_embeddings_f,(liked_embeddings_f.shape[0], model_args.emb_dim))
        true_object_embedding_inv = torch.reshape(liked_embeddings_inv,(liked_embeddings_inv.shape[0], model_args.emb_dim))
    
        d_f = rel_emb[0] * true_object_embedding_f
        d_inv = rel_emb[1] * true_object_embedding_inv
    return (d_f, d_inv)

# return the difference in performance for each upate
def get_diff(x):
    x = x.T
    out = np.empty((x.shape[0] - 1, x.shape[1]))
    for i in range(x.shape[0] - 1):
        out[i] = x[i+1] - x[i]
    return out.T

# make fake item close to gt for testing
def fake_d(gt, rel_emb, model, device, sigma=1):
    while True:
        gt_emb = get_emb(gt, model, device)
        fake_0 = gt_emb[0] + sigma * torch.linalg.norm(gt_emb[0]) * torch.randn(gt_emb[0].shape[0]).to(device)
        fake_1 = gt_emb[1] + sigma * torch.linalg.norm(gt_emb[1]) * torch.randn(gt_emb[1].shape[0]).to(device)
        r = 0.5 * (sim_cos(gt_emb[0], fake_0) + (sim_cos(gt_emb[1], fake_1))) 
        if r > 0.38 and r < 0.52: break
    return (rel_emb[0] * fake_1, rel_emb[1] * fake_0), r.cpu().item() 

def sim_cos(a, b):
    a = torch.squeeze(a)
    b = torch.squeeze(b)
    return (a / torch.linalg.norm(a)) @ (b / torch.linalg.norm(b))

def sim_euc(a, b):
    a = torch.squeeze(a)
    b = torch.squeeze(b)
    return (a) @ (b)

def norm(x):
    (a, b) = x
    a = torch.squeeze(a)
    a = torch.unsqueeze(a / torch.linalg.norm(a), axis=0) 

    b = torch.squeeze(b)
    b = torch.unsqueeze(b / torch.linalg.norm(b), axis=0) 

    return (a, b)

# counts number of similar items to the ground truth

def count_similars(gt_emb, all_embs, likes_emb, sim_metric):
    counter = 0
    for other_emb in all_embs:
        if sim_metric == "cos":
            if sim_cos(likes_emb * gt_emb, likes_emb * other_emb) > 0.9:
                counter +=1
        elif sim_metric == "euc":
            if sim_euc(likes_emb * gt_emb, likes_emb * other_emb) > 0.9:
                counter +=1
    return counter

#  assumes a tuple of tensors and normalizes each row of tensor
def list_norm(x):
    (a, b) = x
    norm = torch.linalg.norm(a, axis=1)
    a = (a.T / norm).T

    norm = torch.linalg.norm(b, axis=1)
    b = (b.T / norm).T
    return (a, b)

#  assumes only 1 row tensor (vector) and normalizes
def single_norm(x):
    a = x / torch.linalg.norm(x)
    return a
# calculates the distance between two vecotrs
def get_distance(a,b):
    a = torch.squeeze(a)
    b = torch.squeeze(b)
    return torch.linalg.norm(a-b)

def scores(a,b):
    for_prod = torch.sum(a[0] * b[0], axis=1)
    rev_prod = torch.sum(a[1] * b[1], axis=1)
    return (for_prod + rev_prod) /2

def MNR_calculator(rank, total_items, all_gt):
    MNR = (total_items - rank) / (total_items - len(all_gt))
    return MNR


# main loop
def critiquing(crit_args, mode):
# setup
#################################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args = Namespace()
    etta = [crit_args.etta_0, crit_args.etta_1, crit_args.etta_2, crit_args.etta_3, crit_args.etta_4]
    alpha = crit_args.alpha

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
    for k, v in vars(crit_args).items():
        save_dict.update({k : str(v)})
    with open(os.path.join(save_path, 'crit hps.yml'), 'w') as f:
        yaml.dump(save_dict, f, sort_keys=False,default_flow_style=False)

    # load model
    model_path = os.path.join(crit_args.load_name, 'models/best_model.pt')
    model = torch.load(model_path).to(device)

    # load dataset + dictionaries
    dataloader = DataLoader(model_args)
    (item_facts_head, item_facts_tail, obj2items, pop_counts) = get_dics(model_args)

    # make arrays with embeddings, and dict to map 
    items_h, items_t, id2index, index2id = get_array(model, dataloader, model_args, device, rec=True)
    item_emb = (items_h, items_t)
    total_items = items_h.shape[0]

    #NORMALIZING
    #item_emb = list_norm(item_emb)

    # get all relationships
    with torch.no_grad():
        rels = dataloader.num_rel
        r = torch.linspace(0, rels - 1, rels).long().to(device)
        rel_f = model.rel_embs(r)
        rel_inv = model.rel_inv_embs(r)

        #NORMALIZING
        # this is weird [num, 2, dim]
        #(rel_f, rel_inv) = list_norm((rel_f, rel_inv))
        rel_emb = torch.stack((rel_f, rel_inv))

        rel_emb = torch.permute(rel_emb, (1,0,2))

    likes_rel = rel_emb[0]

    # all users, gt + data (based on test / train)
    get_gt = GetGT(dataloader.fold, mode)
    data = dataloader.rec_test if mode == 'test' else dataloader.rec_val

    # NOTE: Armin: This was not unique before
    all_users = np.unique(torch.tensor(data[:, 0]).cpu().numpy(), return_counts = False)

    

    # main test loop (each user)
##############################################################
##############################################################
    print('hard coding prior mag(s), n = 1: this must be fixed!')
    print('normalizing embedding vectors')
    rank_track = None
    df_track = None
    dinv_track = None
    score_track = None
    sim_head_track = None
    sim_tail_track = None
    t0 = time.time()
    rec_k = 4 # TODO: this must be an hp
    r_track = []
    for i, user in enumerate(all_users):
        #if i > crit_args.num_users: break
        if i >1: break
        # print('user / sec: {:.3f}'.format(i / (time.time() - t0) ))

        # get ids of top k recs, and all gt from user
        user_emb = get_emb(user, model, device)

        #NORMALIZING 
        #user_emb = norm(user_emb)

        val_or_test_gt, all_gt, train_gt = get_gt.get(user)
        

        # iterature through all gt for single user
        for j, gt in enumerate(val_or_test_gt):
            if j == 2: break
            # get all triplets w gt
            gt_facts = fact_stack(item_facts_head[gt], item_facts_tail[gt])

            # save initial rank and previous user crits 
            sub_track = np.empty(crit_args.session_length + 1)
            sub_track_rank = np.empty(crit_args.session_length + 1)
            sub_track_distance_f = np.empty(crit_args.session_length + 1)
            sub_track_distance_inv = np.empty(crit_args.session_length + 1)
            sub_track_score = np.empty(crit_args.session_length + 1)
            sub_track_sim_head = np.empty(crit_args.session_length + 1)
            sub_track_sim_tail = np.empty(crit_args.session_length + 1)

            ranked = get_scores(user_emb, rel_emb[0], item_emb, model_args.learning_rel)

            rank = get_rank(ranked, [gt], all_gt, id2index) 
            #sub_track[0] = 1 / (rank + 1) 
            #sub_track[0] = rank

            #MNR_pre = MNR_calculator(rank, total_items, all_gt)
            #sub_track[0] = MNR_pre
            #crit = (gt, 0)
            #d = get_d(model, crit, rel_emb, obj2items, crit_args, model_args, device)
            #distance_f = get_distance(user_emb[0], d[0])
            #sub_track[0] = distance_f


            # a few sessions for each user 
            for sn in range(crit_args.session_length):
            ##############################################################
            ##############################################################
                if sn == 0: 
                    update_info = UpdateInfo(user_emb, etta, alpha, crit_args, model_args, device, likes_emb=likes_rel)

                # stack facts, either [-1, rel, tail] or [head, rel, -1]
                rec_ids = [index2id[int(x)] for x in ranked[:rec_k]]
                rec_facts = rec_fact_stack(rec_ids, item_facts_head, item_facts_tail)
                if gt_facts.shape[0] <= 0: continue

                # TESTING: get item most similar to gt in emb space
                real = True
                if real:
                    # NOTE: this used to be return_crit
                    #crit, r = crit_selector(gt, model, item_emb, index2id, device)
                    crit = (gt, 0)
                    gt_ind = id2index[gt]

                    # get d for p(user | d) bayesian update
                    d = get_d(model, crit, rel_emb, obj2items, crit_args, model_args, device)
                    distance_f = get_distance(user_emb[0], d[0])

                    distance_inv = get_distance(user_emb[1], d[1])
                    #print("gt:")
                    #print(gt)
                    #print("gt_ind:")
                    #print(gt_ind)

                    #similar_heads = count_similars(item_emb[0][gt_ind], item_emb[0], likes_rel[1], "cos")
                    #print("similar heads:")
                    #print(similar_heads)
                    #similar_tails = count_similars(item_emb[1][gt_ind], item_emb[1], likes_rel[0], "cos")
                    #print("similar tails:")
                    #print(similar_tails)
                    #print("sn:")
                    #print(sn)
                    #print("pre_rank from get ranked")
                    #print(rank)
                    #print("distance_f")
                    #print(distance_f)
                    #print("distance_inv")
                    #print(distance_inv)
                    #print("norm of user_emb_f:")
                    #print(torch.linalg.norm(user_emb[0]))
                    #print("norm of user_emb_inv:")
                    #print(torch.linalg.norm(user_emb[1]))
                    pre_score = scores(user_emb, d)
                    #print("pre_score:")
                    #print(pre_score)
                    #sub_track[0] = pre_score
                    if sn == 0:
                        sub_track_rank[0] = rank
                        #sub_track_rank[0] = 1/(rank+1)
                        sub_track_distance_f[0] = distance_f
                        sub_track_distance_inv[0] = distance_inv
                        sub_track_score[0] = pre_score
                        #sub_track_sim_head[0] = similar_heads
                        #sub_track_sim_tail[0] = similar_tails

                    update_info.store(d=d, crit_rel_emb=rel_emb[crit[1]]) # crit[1]
                else: 
                    d, r = fake_d(gt, rel_emb[0], model, device, sigma=1.5)
                    update_info.store(d=d, crit_rel_emb=rel_emb[0])
                r_track.append(r)

                # perform update
                if crit_args.evidence_type == 'direct':
                    beta_update(update_info, sn, crit_args, model_args, device, crit_args.update_type, crit_args.map_finder, etta, alpha)
                if crit_args.evidence_type == 'indirect':
                    beta_update_indirect(update_info, sn, crit_args, model_args, device, crit_args.update_type, crit_args.map_finder, etta, alpha)

                # track rank in training
                new_user_emb, _ = update_info.get_priorinfo()
                #NORMALIZING
                #new_user_emb = norm(new_user_emb)
                #print("norm of new_user_emb_f:")
                #print(torch.linalg.norm(new_user_emb[0]))
                #print("norm of new_user_emb_inv:")
                #print(torch.linalg.norm(new_user_emb[1]))
                ranked = get_scores(new_user_emb, rel_emb[0], item_emb, model_args.learning_rel)
                post_rank = get_rank(ranked, [gt], all_gt, id2index)
                #print("post rank")
                #print(post_rank)
                post_distance_f = get_distance(new_user_emb[0], d[0])
                post_distance_inv = get_distance(new_user_emb[1], d[1])
                #print("post distance_f")
                #print(post_distance_f)
                #print("post distance_inv")
                #print(post_distance_inv)
                #print("post score:")
                post_score = scores(new_user_emb, d)
                #print(post_score)

                #sub_track_rank[sn + 1] = 1 / (post_rank + 1)
                sub_track_rank[sn + 1] = post_rank
                #sub_track[sn + 1] = 1 / (post_rank + 1)
                sub_track_distance_f[sn+1] = post_distance_f
                sub_track_distance_inv[sn+1] = post_distance_inv
                sub_track_score[sn+1] = post_score
                #sub_track_sim_head[sn+1] = similar_heads
                #sub_track_sim_tail[sn+1] = similar_tails
              
                #sub_track[sn + 1] = 1 / (post_rank + 1)
                #sub_track[sn + 1] = post_rank
                #sub_track[sn+1] = post_score

                #MNR_post = MNR_calculator(post_rank, total_items, all_gt)
                #sub_track[sn + 1] = MNR_post
                #sub_track[sn + 1] = post_distance_f

                #print("sub_track")
                #print(sub_track)


            # update w new data
            sub_track_rank = np.expand_dims(sub_track_rank, axis=0)
            sub_track_distance_f = np.expand_dims(sub_track_distance_f, axis=0)
            sub_track_distance_inv = np.expand_dims(sub_track_distance_inv, axis=0)
            sub_track_score = np.expand_dims(sub_track_score, axis=0)
            #sub_track_sim_head = np.expand_dims(sub_track_sim_head, axis=0)
            #sub_track_sim_tail = np.expand_dims(sub_track_sim_tail, axis=0)
            if rank_track is None: rank_track = sub_track_rank
            else: rank_track = np.concatenate((rank_track, sub_track_rank ))
            if df_track is None: df_track = sub_track_distance_f
            else: df_track = np.concatenate((df_track, sub_track_distance_f ))
            if dinv_track is None: dinv_track = sub_track_distance_inv
            else: dinv_track = np.concatenate((dinv_track, sub_track_distance_inv ))
            if score_track is None: score_track = sub_track_score
            else: score_track = np.concatenate((score_track, sub_track_score))
            #if sim_head_track is None: sim_head_track = sub_track_sim_head
            #else: sim_head_track = np.concatenate((sim_head_track, sub_track_sim_head))
            #if sim_tail_track is None: sim_tail_track = sub_track_sim_tail
            #else: sim_tail_track = np.concatenate((sim_tail_track, sub_track_sim_tail))
            #st = np.expand_dims(sub_track, axis=0)
            #print("st:")
            #print(st)
            #if rank_track is None: rank_track = st
            #else: rank_track = np.concatenate((rank_track, st))
            #print("rank_track")
            #print(rank_track)
    #sys.exit()
       
    # plotting
    print("plotting")
    #print(np.mean(r_track), np.std(r_track))


    print("rank_Track")
    print(rank_track)
    print("sim head track")
    print(sim_head_track)
    #for j in range(sim_head_track.shape[0]):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(141) 
    ax2 = fig.add_subplot(142) 
    ax3 = fig.add_subplot(143) 
    ax4 = fig.add_subplot(144)
    for (data, ax) in [(rank_track, ax1), (df_track, ax2), (dinv_track, ax3), (score_track, ax4)]:
        x_ = np.arange(data.shape[1])
        m = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        ax.errorbar(x_, m, std)  
        #ax.plot(x_, data[j,:])
    #ax2.set_title('{} similar heads and {} similar tails'.format(sim_head_track[j][0],sim_tail_track[j][0]))
    #ax1.set_ylabel('Rank')
    ax1.set_ylabel('Rank')
    ax2.set_ylabel('Distance (forward)')
    ax3.set_ylabel('Distance (inverse)')
    ax4.set_ylabel('Score')
    plt.tight_layout() 
    plt.savefig(os.path.join(save_path, 'debug.jpg'))
    #plt.savefig(os.path.join(save_path, 'debug{}.jpg'.format(j)))
    plt.show()
        
    sys.exit()

    #fig = plt.figure(figsize=(10,5))
    #ax1 = fig.add_subplot(121)  
    #ax2 = fig.add_subplot(122)  

    #for (data, ax) in [(rank_track, ax1), (get_diff(rank_track), ax2)]:
        #m = np.mean(data, axis=0)
        #std = np.std(data, axis=0)
        #x_ = np.arange(m.shape[0])
        #ax.errorbar(x_, m, std)  


    #ax1.set_title('Score')
    #ax2.set_title('$\Delta$ Score')
    #ax2.axhline(0, color='r')
    #plt.tight_layout() 
    #plt.savefig(os.path.join(save_path, 'debug.jpg'))
    #plt.show()
    #sys.exit()

    # save results
    if mode == 'val':
        mrr = save_metrics_critiquing(rank_track, crit_args.test_name, mode)
        return mrr
    else:
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
