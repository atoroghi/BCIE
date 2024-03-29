import os, sys, torch, pickle, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('..')
from tester import get_rank, get_rank_nongt

# for tracking and saving important debug info
class InfoTrack:
    def __init__(self, sess_len, objective, param_tuning, session):
        self.sess_len = sess_len
        self.objective = objective
        self.dists = []
        self.ranks = []
        self.scores = []
        self.pcds = []
        self.user_embs_for = []
        self.user_embs_inv = []
        self.user_embs_prec_for = []
        self.user_embs_prec_inv = []
        self.gts = []
        self.crit_nodes = []
        self.crit_rels = []
        self.users = []
        self.param_tuning = param_tuning
        self.session = session

    # make new storage units (append to list later)
    def new_temps(self):
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.d_temp = np.zeros(self.sess_len)
            self.r_temp = np.zeros(self.sess_len + 1)
            self.s_temp = np.zeros(self.sess_len + 1)
            self.pcds_temp = np.zeros(self.sess_len + 1)
            self.users_temp = np.zeros(self.sess_len + 1)
            self.user_embs_for_temp = torch.zeros((1, self.sess_len + 1)).to(self.device)
            # increse its dim. user_emb isn't a number but a tensor
            self.user_embs_for_temp = self.user_embs_for_temp[:,:, None]
            self.user_embs_inv_temp = torch.zeros((1, self.sess_len + 1)).to(self.device)
            self.user_embs_inv_temp = self.user_embs_inv_temp[:,:, None]
            self.user_embs_prec_for_temp = torch.zeros((self.sess_len + 1)).to(self.device)
            self.user_embs_prec_for_temp = self.user_embs_prec_for_temp[ :, None, None]
            self.user_embs_prec_inv_temp = torch.zeros((self.sess_len + 1)).to(self.device)
            self.user_embs_prec_inv_temp = self.user_embs_prec_inv_temp[ :, None, None]
            self.gt_temp = np.zeros(self.sess_len + 1)
            self.crit_node_temp = np.zeros(self.sess_len + 1)
            self.crit_rel_temp = np.zeros(self.sess_len + 1)

    # calculate scores
    def calc_score(self, dist):
        user = dist[0]
        d = dist[1]
        s_for = torch.sum(user[0] * d[0])
        s_back = torch.sum(user[1] * d[1])
        return 0.5*(s_for + s_back)

    # update info into arrays
    # NOTE: this class is very brittle...
    def store(self, sess_no, rank=None, score=None, dist=None, pcd=None, user_emb=None, user_emb_prec=None, gt=None, crit=None, user=None):
        if sess_no == 0: self.new_temps()
        if rank is not None: self.r_temp[sess_no:] = rank
        if score is not None: self.s_temp[sess_no] = score
        if pcd is not None: self.pcds_temp[sess_no:] = pcd
        if dist is not None: self.d_temp[sess_no-1] = self.calc_score(dist)
        if user is not None: self.users_temp[sess_no:] = user
        if user_emb is not None:
            self.user_embs_for_temp = self.user_embs_for_temp.expand(-1,-1,user_emb[0].shape[1]).clone()
            self.user_embs_for_temp[0][sess_no] = user_emb[0]
            self.user_embs_inv_temp = self.user_embs_inv_temp.expand(-1,-1,user_emb[1].shape[1]).clone()
            #1x6x14
            self.user_embs_inv_temp[0][sess_no] = user_emb[1]
        if user_emb_prec is not None:
            self.user_embs_prec_for_temp  = self.user_embs_prec_for_temp.expand(-1,user_emb_prec[0].shape[1], user_emb_prec[0].shape[1]).clone()
            #6x14x14
            self.user_embs_prec_for_temp[sess_no:] = user_emb_prec[0]
            self.user_embs_prec_inv_temp  = self.user_embs_prec_inv_temp.expand(-1,user_emb_prec[1].shape[1], user_emb_prec[1].shape[1]).clone()
            self.user_embs_prec_inv_temp[sess_no:] = user_emb_prec[1]
        if gt is not None: self.gt_temp[sess_no:] = gt
        if crit is not None:
            self.crit_node_temp[sess_no:]= crit[0]
            self.crit_rel_temp[sess_no:]= crit[1]
        self.session = sess_no

        # list of each crit sesh
        #if sess_no == self.sess_len:
        self.dists.append(self.d_temp)
        self.ranks.append(self.r_temp)
        self.scores.append(self.s_temp)
        self.pcds.append(self.pcds_temp)
        self.user_embs_for.append(self.user_embs_for_temp)
        self.user_embs_inv.append(self.user_embs_inv_temp)
        self.user_embs_prec_for.append(self.user_embs_prec_for_temp)
        self.user_embs_prec_inv.append(self.user_embs_prec_inv_temp)
        self.gts.append(self.gt_temp)
        self.crit_nodes.append(self.crit_node_temp)
        self.crit_rels.append(self.crit_rel_temp)
        self.users.append(self.users_temp)



    # save info for stopping et al.
    def save(self, test_name):
        # convert lists to numpy arrays
        dists = np.array(self.dists)
        ranks = np.array(self.ranks)
        scores = np.array(self.scores)
        pcds = np.array(self.pcds)
        #user_embs = np.array(self.user_embs)
        #user_embs_prec = np.array(self.user_embs_prec)
        user_embs_for = self.user_embs_for
        user_embs_inv = self.user_embs_inv
        user_embs_prec_for = self.user_embs_prec_for
        user_embs_prec_inv = self.user_embs_prec_inv
        gts = np.array(self.gts)
        crit_nodes = np.array(self.crit_nodes)
        crit_rels = np.array(self.crit_rels)
        users = np.array(self.users)
        #save_path = os.path.join('results', test_name)
        save_path = test_name
        os.makedirs(save_path, exist_ok=True)

        # reduce track, save single number in stop_metric.npy
        #mrr_last = np.mean(1 / (ranks[:, -1] + 1))
        #mrr_last = (np.mean(ranks[:,0]) - np.mean(ranks[:,-1]))
        if self.param_tuning == 'per_session':
            mrr_last = (np.mean(ranks[:,self.session-1]) - np.mean(ranks[:,self.session]))
            #hr_last = (np.sum(ranks[:,-1]<12, axis = 0) - np.sum(ranks[:,0]<12, axis = 0)) / (ranks[:,-1].shape[0])
            hr_last = (np.sum(ranks[:,self.session]<12, axis = 0) - np.sum(ranks[:,self.session-1]<12, axis = 0)) / (ranks[:,self.session].shape[0])
            pcd_last = np.mean(pcds[:,self.session])    
        elif self.param_tuning == 'together':
            mrr_last = (np.mean(ranks[:,0]) - np.mean(ranks[:,-1]))
            temp = 0
            for i in range(1,self.sess_len+1):
                temp += np.sum(ranks[:,i]<12, axis = 0) - np.sum(ranks[:,0]<12, axis = 0)
            hr_last = temp / (self.sess_len * ranks[:,0].shape[0]) 
            #hr_last = (np.sum(ranks[:,self.sess_len]<12, axis = 0) - np.sum(ranks[:,0]<12, axis = 0))/ (ranks[:,0].shape[0]) 
            pcd_last = np.mean(pcds[:,self.session])    
        # take this and plot it, look at it etc...
        if self.objective == 'hits':
            print("last hit rate:")
            print((np.sum(ranks[:,-1]<12, axis = 0)/ (ranks[:,self.session].shape[0])))
            np.save(os.path.join(save_path, 'stop_metric.npy'), hr_last)
            with open(os.path.join(save_path,'stop_metric.txt'), 'w') as f:
                f.write(str(hr_last))
 
        elif self.objective == 'rank':
            print("last rank average:")
            print(np.mean(ranks[:,-1]))
            np.save(os.path.join(save_path, 'stop_metric.npy'), mrr_last)
            with open(os.path.join(save_path,'stop_metric.txt'), 'w') as f:
                f.write(str(mrr_last))
        elif self.objective == 'pcd':
            print("PCD value:")
            print(np.mean(pcds[:,self.session]))
            np.save(os.path.join(save_path, 'stop_metric.npy'), pcd_last)
            with open(os.path.join(save_path,'stop_metric.txt'), 'w') as f:
                f.write(str(pcd_last))
        np.save(os.path.join(save_path, 'rank_track.npy'), ranks)
        np.save(os.path.join(save_path, 'score_track.npy'), scores)
        np.save(os.path.join(save_path, 'dist_track.npy'), dists)
        #np.save(os.path.join((save_path, 'user_embs.npy'), user_embs))
        #np.save(os.path.join((save_path, 'user_embs_prec.npy'), user_embs_prec))

        #torch.save(user_embs_for, os.path.join(save_path, 'user_embs_for.pt'))
        #torch.save(user_embs_inv, os.path.join(save_path, 'user_embs_inv.pt'))
        #torch.save(user_embs_prec_for, os.path.join(save_path, 'user_embs_prec_for.pt'))
        #torch.save(user_embs_prec_inv, os.path.join(save_path, 'user_embs_prec_inv.pt'))
        #np.save(os.path.join(save_path, 'gts.npy'), gts)
        #np.save(os.path.join(save_path, 'crit_nodes.npy'), crit_nodes)
        #np.save(os.path.join(save_path, 'crit_rels.npy'), crit_rels)
        #np.save(os.path.join(save_path, 'users.npy'), users)

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
        ax5 = fig.add_subplot(131)
        ax6 = fig.add_subplot(132)
        ax7 = fig.add_subplot(133)
        #ax5.set_yscale('log')
        hits_10 = np.sum(ranks<12 , axis =0) / (ranks[:,0].shape[0])
        ax5.errorbar(np.arange(hits_10.shape[0]), hits_10, linestyle='--', fmt = 'o')
        ax5.set_xlabel('Step', fontsize=16)
        ax5.set_ylabel('Hit Rate @ 10', fontsize=16)




        
        hits_5 = np.sum(ranks<7 , axis =0) / (ranks[:,0].shape[0])
        ax6.errorbar(np.arange(hits_5.shape[0]), hits_5, linestyle='--', fmt = 'o')
        ax6.set_xlabel('Step', fontsize=16)
        ax6.set_ylabel('Hit Rate @ 5', fontsize=16)

        
        hits_20 = np.sum(ranks<22 , axis =0) / (ranks[:,0].shape[0])
        ax7.errorbar(np.arange(hits_20.shape[0]), hits_20, linestyle='--', fmt = 'o')
        ax7.set_xlabel('Step', fontsize=16)
        ax7.set_ylabel('Hit Rate @ 20', fontsize=16)
        plt.savefig(os.path.join(save_path, 'HR20.png'))
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
def get_d(model, crit, rel_emb, obj2items, get_emb, crit_args, model_args, device):
    (crit_node, crit_rel) = crit
    rel_emb = rel_emb[crit_rel]

    # single embedding of crit node
    if crit_args.critique_target == 'single':
        node_emb = get_emb(crit_node, model, device)

        if crit_args.evidence_type == 'direct':
            d_f = rel_emb[0] * node_emb[1]
            d_inv = rel_emb[1] * node_emb[0]
        else: 
            d_f = node_emb[1]
            d_inv = node_emb[0]

    # make stack of likes * items related to feedback node
    elif crit_args.critique_target == 'multi':
        liked_items = obj2items[crit_node]
        #print("mapped items:")
        #print(liked_items)
        #liked_items = np.random.permutation(liked_items)

        # get and stack things
        liked_embeddings_list_f = []
        liked_embeddings_list_inv = []

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



# Used for the Single Step experiment to calculate postcritiquingdiff

def get_postdiff(ranked_pre, ranked_post, crit_node, obj2items, all_gt, id2index):
    # items that are related to the crit node
    pos_items = obj2items[crit_node]

    
    # some elements in the obj2item dict are not items. we need to remove them
    pos_items = pos_items[np.isin(pos_items, np.fromiter(id2index.keys(), dtype=int))]
    
    
    
    #for item in pos_items:
     #   if item not in id2index.keys():
      #      pos_items = np.setdiff1d(pos_items, item)

    # avg rank of positive items before critique
    try:
        
        pos_ranked_pre = np.mean(get_rank(ranked_pre, pos_items, all_gt, id2index))
        
        
        
    
    except:
        
        pos_ranked_pre = np.mean(get_rank_nongt(ranked_pre, pos_items, all_gt, id2index))
        
    
     # avg rank of positive items after critique
    try:
        
        pos_ranked_post = np.mean(get_rank(ranked_post, pos_items, all_gt, id2index))

    except:
        pos_ranked_post = np.mean(get_rank_nongt(ranked_post, pos_items, all_gt, id2index))

    pcd = (1 + pos_ranked_pre - pos_ranked_post) / (pos_ranked_pre+1)
    

    return pcd


