import os, sys, torch, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# for tracking and saving important debug info
class InfoTrack:
    def __init__(self, sess_len, objective, param_tuning, session):
        self.sess_len = sess_len
        self.objective = objective
        self.dists = []
        self.ranks = []
        self.scores = []
        self.param_tuning = param_tuning
        self.session = session

    # make new storage units (append to list later)
    def new_temps(self):
            self.d_temp = np.zeros(self.sess_len)
            self.r_temp = np.zeros(self.sess_len + 1)
            self.s_temp = np.zeros(self.sess_len + 1)

    # calculate scores
    def calc_score(self, dist):
        user = dist[0]
        d = dist[1]
        s_for = torch.sum(user[0] * d[0])
        s_back = torch.sum(user[1] * d[1])
        return 0.5*(s_for + s_back)

    # update info into arrays
    # NOTE: this class is very brittle...
    def store(self, sess_no, rank=None, score=None, dist=None):
        if sess_no == 0: self.new_temps()
        if rank is not None: self.r_temp[sess_no:] = rank
        if score is not None: self.s_temp[sess_no] = score
        if dist is not None: self.d_temp[sess_no-1] = self.calc_score(dist)
        self.session = sess_no

        # list of each crit sesh
        #if sess_no == self.sess_len:
        self.dists.append(self.d_temp)
        self.ranks.append(self.r_temp)
        self.scores.append(self.s_temp)


    # save info for stopping et al.
    def save(self, test_name):
        # convert lists to numpy arrays
        dists = np.array(self.dists)
        ranks = np.array(self.ranks)
        scores = np.array(self.scores)
        #save_path = os.path.join('results', test_name)
        save_path = test_name
        os.makedirs(save_path, exist_ok=True)

        # TODO: is this a good metric?
        # reduce track, save single number in stop_metric.npy
        #mrr_last = np.mean(1 / (ranks[:, -1] + 1))
        #mrr_last = (np.mean(ranks[:,0]) - np.mean(ranks[:,-1]))
        if self.param_tuning == 'per_session':
            mrr_last = (np.mean(ranks[:,self.session-1]) - np.mean(ranks[:,self.session]))
            #hr_last = (np.sum(ranks[:,-1]<12, axis = 0) - np.sum(ranks[:,0]<12, axis = 0)) / (ranks[:,-1].shape[0])
            hr_last = (np.sum(ranks[:,self.session]<12, axis = 0) - np.sum(ranks[:,self.session-1]<12, axis = 0)) / (ranks[:,self.session].shape[0])    
        elif self.param_tuning == 'together':
            mrr_last = (np.mean(ranks[:,-1]) - np.mean(ranks[:,0]))
            temp = 0
            for i in range(1,self.sess_len):
                temp += np.sum(ranks[:,i]<12, axis = 0) - np.sum(ranks[:,0]<12, axis = 0)
            hr_last = temp / (self.sess_len * ranks[:,0].shape[0])    
        # take this and plot it, look at it etc...
        if self.objective == 'hits':
            print("last hit rate:")
            print((np.sum(ranks[:,self.session]<12, axis = 0)/ (ranks[:,self.session].shape[0])))
            np.save(os.path.join(save_path, 'stop_metric.npy'), hr_last)
        elif self.objective == 'ranks':
            print("last rank average:")
            print(np.mean(ranks[:,-1]))
            np.save(os.path.join(save_path, 'stop_metric.npy'), mrr_last)
        np.save(os.path.join(save_path, 'rank_track.npy'), ranks)
        np.save(os.path.join(save_path, 'score_track.npy'), scores)
        np.save(os.path.join(save_path, 'dist_track.npy'), dists)

        # plotting
        sns.set_theme()
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.set_yscale('log')

        n = 20
        ax1.plot(ranks.T[:,:n]) 
        ax2.plot(get_diff(ranks).T[:,:n]) 
        ax3.plot(scores.T[:,:n]) 

        ax1.set_title('Rank')
        ax2.set_title('$\Delta$ Rank')
        ax3.set_title('Scores')

        plt.savefig(os.path.join(save_path, 'info.png'))
        #sys.exit()

        fig = plt.figure(figsize=(12,8), dpi=300)
        ax4 = fig.add_subplot(111)
        ax4.set_yscale('log')
        ranks_mean = np.mean(ranks, axis=0)
        yerr_ranks = 1.96 / np.sqrt(ranks.shape[0]) * np.std(ranks, axis=0)
        ax4.errorbar(np.arange(ranks.shape[1]), ranks_mean, yerr = yerr_ranks, linestyle='--', fmt = 'o')
        ax4.set_xlabel('Step', fontsize=16)
        ax4.set_ylabel('Average Rank', fontsize=16)
        plt.savefig(os.path.join(save_path, 'AvgRank.png'))


        # plotting hitrate@10
        fig = plt.figure(figsize=(12,8), dpi=300)
        ax5 = fig.add_subplot(111)
        #ax5.set_yscale('log')
        hits_10 = np.sum(ranks<12 , axis =0) / (ranks[:,5].shape[0])
        ax5.errorbar(np.arange(hits_10.shape[0]), hits_10, linestyle='--', fmt = 'o')
        ax5.set_xlabel('Step', fontsize=16)
        ax5.set_ylabel('Hit Rate @ 10', fontsize=16)
        plt.savefig(os.path.join(save_path, 'HR10.png'))

        # same plot but instead of CIs, errorbars are STD

        #fig = plt.figure(figsize=(12,8), dpi=300)
        #ax6 = fig.add_subplot(111)
        #ax6.set_yscale('log')
        #ax6.errorbar(np.arange(ranks.shape[1]), ranks_mean, yerr = np.std(ranks, axis=0), linestyle='--', fmt = '-o')
        #ax6.set_xlabel('Step', fontsize=16)
        #ax6.set_ylabel('Average Rank', fontsize=16)
        #plt.savefig(os.path.join(save_path, 'AvgRank(STD).png'))


        

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

# TODO: what does this do again?
def rec_fact_stack(ids, items_facts_head, items_facts_tail):
    rec_facts = []
    for rec_id in ids:
        rec_facts.append(fact_stack(items_facts_head[rec_id], items_facts_tail[rec_id]))
    return rec_facts

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

        # TODO: resolve this
        #if crit_args.evidence_type == 'indirect':
        # Armin: shouldn't this be precision? so 1/reg_lambda ? 
        self.z_prec_f = model_args.reg_lambda * np.eye(model_args.emb_dim)
        self.z_prec_inv = model_args.reg_lambda * np.eye(model_args.emb_dim)
        self.z_mean_f = np.zeros(model_args.emb_dim)
        self.z_mean_inv = np.zeros(model_args.emb_dim)

# get d embedding, used for p(u | d) baysian update
def get_d(model, crit, rel_emb, obj2items, get_emb, crit_args, model_args, device):
    (crit_node, crit_rel) = crit
    rel_emb = rel_emb[crit_rel]

    # single embedding of crit node
    if crit_args.critique_target == 'single':
        node_emb = get_emb(crit_node, model, device)

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
        #print("mapped items:")
        #print(liked_items)
        #liked_items = np.random.permutation(liked_items)

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

# make fake item close to gt for testing
def fake_d(gt, rel_emb, model, device, sigma=1):
    while True:
        gt_emb = get_emb(gt, model, device)
        fake_0 = gt_emb[0] + sigma * torch.linalg.norm(gt_emb[0]) * torch.randn(gt_emb[0].shape[0]).to(device)
        fake_1 = gt_emb[1] + sigma * torch.linalg.norm(gt_emb[1]) * torch.randn(gt_emb[1].shape[0]).to(device)
        r = 0.5 * (sim_cos(gt_emb[0], fake_0) + (sim_cos(gt_emb[1], fake_1))) 
        if r > 0.38 and r < 0.52: break
    return (rel_emb[0] * fake_1, rel_emb[1] * fake_0), r.cpu().item() 

# return the difference in performance for each upate
def get_diff(x):
    x = x.T
    out = np.empty((x.shape[0] - 1, x.shape[1]))
    for i in range(x.shape[0] - 1):
        out[i] = x[i+1] - x[i]
    return out.T

###########################################################3
# useful debug functions
###########################################################3
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

# gets similarity of the most similar item embedding
def most_similar(all_embs, likes_emb, sim_metric):
    most_similars = []
    for candidate_emb in all_embs:
        if sim_metric == "euc":
            euc_sims = torch.sum(candidate_emb * all_embs, dim = 1)
            max_sim = torch.max(euc_sims)
            most_similars.append(max_sim.item())

        elif sim_metric == "cos":

            normalized_candidate = candidate_emb / torch.linalg.norm(candidate_emb)
            all_embs_norm = torch.linalg.norm(all_embs, axis=1)
            normalized_all_embs = (all_embs.T / all_embs_norm).T
            cos_sims = torch.sum(normalized_candidate * normalized_all_embs, dim = 1)
            max_sim = torch.topk(cos_sims, k=2)[0][1]
            most_similars.append(max_sim.item())

    return np.mean(most_similars) , np.std(most_similars)




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

# we may want to use these again...
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

# to numpy array
def unpack_dic(dic, ids):
    array = None
    for i in ids:
        out = np.array(dic[i])
        if array is None: array = out
        else: array = np.vstack((array, out))
    return array

def sim_selector(gt, item_emb, id2index, index2id, device, k=1):
    # get embeddings
    gt_head = item_emb[0][id2index[gt]]
    gt_tail = item_emb[1][id2index[gt]]

    # dot product and add
    forw = torch.sum(gt_head * item_emb[0], axis=1)
    back = torch.sum(gt_tail * item_emb[1], axis=1)
    both = forw + back

    # get top item that isn't gt
    spot = torch.topk(both, k=k+1)[1]
    pick = [index2id[spot[i].cpu().item()] for i in range(k+1)]
    if gt in pick: pick.remove(gt)
    ind = np.random.randint(len(pick))
    return (pick[ind], 0)
