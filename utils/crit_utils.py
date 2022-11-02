import os, sys, torch, pickle
import numpy as np
import matplotlib.pyplot as plt

# for tracking and saving important debug info
class InfoTrack:
    def __init__(self, sess_len):
        self.sess_len = sess_len
        self.dists = []
        self.ranks = []
        self.scores = []

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
        if rank is not None: self.r_temp[sess_no] = rank
        if score is not None: self.s_temp[sess_no] = score
        if dist is not None: self.d_temp[sess_no-1] = self.calc_score(dist)

        # list of each crit sesh
        if sess_no == self.sess_len:
            self.dists.append(self.d_temp)
            self.ranks.append(self.r_temp)
            self.scores.append(self.s_temp)

    # save info for stopping et al.
    def save(self, test_name):
        # convert lists to numpy arrays
        dists = np.array(self.dists)
        ranks = np.array(self.ranks)
        scores = np.array(self.scores)
        print(dists.shape, ranks.shape, scores.shape)
        
        save_path = os.path.join('results', test_name)
        os.makedirs(save_path, exist_ok=True)

        # TODO: is this a good metric?
        # reduce track, save single number in stop_metric.npy
        improv = ranks[:, -1] - ranks[:, 0]
        
        # take this and plot it, look at it etc...
        y = np.mean(improv)
        np.save(os.path.join(save_path, 'stop_metric.npy'), y)
        np.save(os.path.join(save_path, 'rank_track.npy'), ranks)
        np.save(os.path.join(save_path, 'score_track.npy'), scores)
        np.save(os.path.join(save_path, 'dist_track.npy'), dists)

        # plotting
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_yscale('log')

        #ax2.plot(np.mean(get_diff(ranks), axis=0)) 
        #ax3.plot(np.mean(dists, axis=0)) 
        #ax4.plot(np.mean(scores, axis=0)) 

        ax1.plot(ranks.T) 
        ax2.plot(get_diff(ranks).T) 
        ax3.plot(dists.T) 
        ax4.plot(scores.T) 


        ax1.set_title('Rank')
        ax2.set_title('$\Delta$ Rank')
        ax3.set_title('Distances')
        ax4.set_title('Scores')

        plt.savefig(os.path.join(save_path, 'info.png'))
        sys.exit()

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