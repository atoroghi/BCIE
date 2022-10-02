from cmath import tau
import os, pickle, sys, argparse, yaml
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from updater import Updater
from updater_indirect import Updater_Indirect
from dataload import DataLoader
from tester import get_array, get_emb, get_scores, get_rank, GetGT
from recommender import select_critique, remove_chosen_critiques, obj2item
from utils.plots import RankTrack, rank_plot, save_metrics_critiquing
from launch import get_model_args

def get_args_critique():
    # TODO: load all hp's from load_name
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help='name of folder where results are saved')
    parser.add_argument('-load_name', default='dev', type=str, help='name of folder where model is')
    #parser.add_argument('-alpha', default=0.01, type=float, help='Learning rate for Laplace Approximation')
    parser.add_argument('-fold', default=0, type=int, help='fold number')

    #TODO: Have multiple ettas for each session
    #TODO: This is bad (list)
    parser.add_argument('-user_prec', default=1e5, type=float, help='prior precision')
    parser.add_argument('-ettaone', default=1, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-ettatwo', default=1, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-ettathree', default=1, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-ettafour', default=1, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-ettafive', default=1, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-critique_target', default='item', type=str, help='object or item')
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    parser.add_argument('-update_type', default='laplace', type=str, help='laplace or gaussian')
    parser.add_argument('-critique_mode', default='random', type=str, help='random or pop or diff')
    parser.add_argument('-likelihood_precision', default=1e2, type=float, help='likelihood precision')

    # to remove
    # TODO: this is very bad and ugly... never do this
    parser.add_argument('-num_users', default=100, type=int, help='number of users')
    #parser.add_argument('-model_type', default=' ', type=str, help='this is bad')

    args = parser.parse_args()
    return args

# get dics w info about all triplets
def get_dics(args):
    data_path = 'datasets/' + args.dataset
    # all triples when item is head
    with open(os.path.join(data_path, 'items_facts_head.pkl'), 'rb') as f:
        items_facts_head = pickle.load(f)
    # all triples when item is tail
    with open(os.path.join(data_path, 'items_facts_tail.pkl'), 'rb') as f:
        items_facts_tail = pickle.load(f)
    # what is this???
    # This dict maps objects (non-item entities) to items, e.g., "Bergman" to "Saraband", "Wild Strawberries", etc
    with open(os.path.join(data_path, 'obj2items.pkl'), 'rb') as f:
        obj2items = pickle.load(f)
    # total count for each node
    with open(os.path.join(data_path, 'pop_counts.pkl'), 'rb') as f:
        pop_counts = pickle.load(f)
    return items_facts_head, items_facts_tail, obj2items, pop_counts

# return info about each of the priors
class Priors:
    #TODO: this is for more complex prior
    def __init__(self, crit_args, model_args):
        # we assume this (this is a hp)
        self.user_prec_f = crit_args.user_prec * np.eye(model_args.emb_dim)
        self.user_prec_inv = crit_args.user_prec * np.eye(model_args.emb_dim)

        # the model defines this. N(0, lambda*I)
        # prior over items for I^2
        if args.evidence_type == 'indirect':
            self.z_prec_f = model_args.reg_lambda * np.eye(model_args.emb_dim)
            self.z_prec_inv = model_args.reg_lambda * np.eye(model_args.emb_dim)
            self.z_mean_f = np.zeros(model_args.emb_dim)
            self.z_mean_inv = np.zeros(model_args.emb_dim)

# main loop
def critiquing(model, crit_args, mode):
    model_args = get_model_args() 
    priors = Priors(crit_args, model_args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # get all dictionaries
    etta_dict = {1: args.ettaone, 2: args.ettatwo, 3: args.ettathree, 4: args.ettafour, 5: args.ettafive}
    (item_facts_head, item_facts_tail, obj2items, pop_counts) = get_dics(model_args)

    # load model and get parameter from file
    path = os.path.join('results', args.load_name, 'fold_{}'.format(args.fold))
    with open(os.path.join(path, 'info.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        for key in yml.keys():
            if key != "test_name":
                setattr(model_args, key, yml[key])
    
    # print important hps
    print('alpha: {}\tettaone: {}\tettatwo: {}\tettathree: {}\tettafour: {}\tcritique mode: {}\tcritique target: {}'.format(
        crit_args.alpha, crit_args.ettaone, crit_args.ettatwo, crit_args.ettathree, crit_args.ettafour, crit_args.critique_mode, crit_args.critique_target))

    # load datasets
    print('loading dataset: {}'.format(model_args.dataset))
    dataloader = DataLoader(model_args)

    # make arrays with embeddings, and dict to map 
    rec_h, rec_t, id2index, index2id = get_array(model, dataloader, model_args, rec=True)
    item_emb = (rec_h, rec_t)

    # load gt info and track rank scores classes
    get_gt = GetGT(dataloader.fold, mode)

    # get all relationships
    with torch.no_grad():
        rels = dataloader.num_rel
        r = torch.linspace(0, rels - 1, rels).long().to('cuda')
        rel_f = model.rel_embs(r)
        rel_inv = model.rel_inv_embs(r)
        rel_emb = torch.stack((rel_f, rel_inv))
        rel_emb = torch.permute(rel_emb, (1,0,2))

    # all users, gt selects test / train gts
    data = dataloader.rec_test if mode == 'test' else dataloader.rec_val

    #shouldn't we just enumerate ul_test keys?
    all_users = torch.tensor(data[:2, 0]).to('cuda')
    rank_track = RankTrack()
    rank_track = np.empty((1, args.session_length + 1))

    # TODO: how much of this is on the cpu???
    # main test loop
    for i, user in tqdm(enumerate(all_users)):
        # TODO: abstract this process in tester and call one function
        user_emb = get_emb(user, model)
        ranked = get_scores(user_emb, rel_emb[0], item_emb, dataloader, model_args.learning_rel)
        recommended_ids = [index2id[int(x)] for x in ranked[:20]]
        test_gt, all_gt, train_gt = get_gt.get(user.cpu().item())

        # TODO: check case where use doesn't have 20 liked items? 
        # iterature through all gt
        rank_track_user = np.empty((len(test_gt), args.session_length + 1))
        for j, gt in enumerate(test_gt):
            pre_rank = get_rank(ranked, [gt], all_gt, id2index)
            rank_track_user[j,0] = pre_rank

            # history of user critiques (no repeats) 
            prev_crit = np.array([[0, 0]])
            ht_facts = np.vstack([item_facts_head[gt], item_facts_tail[gt]])
            # TODO: check case where we have enough data
            # remove 1 each time, we should have more than 5?

            # TODO: this is too many lines!!
            for session_no in range(1, args.session_length+1):
                # initialize prior for first critique
                if session_no == 0:
                    priors = Priors(crit_args, model_args)
                # remove prev_crit from ht_facts 
                if session_no > 0:

                    dims = np.maximum(prev_crit.max(0), ht_facts.max(0)) + 1
                    ht_facts = ht_facts[~np.in1d(np.ravel_multi_index(ht_facts.T, dims), np.ravel_multi_index(prev_crit.T, dims))]          

                # TODO: if this is slow? maybe fix
                # get all tails to recover head / tail info about crit gt
                items_facts_tail_gt = np.hstack([item_facts_tail[gt], np.full((np.shape(item_facts_tail[gt])[0], 1), -1)])

                # TODO: make this 2 lines 
                # get all pairs
                # put them all in a stack

                rec_facts_head = np.array([])   
                rec_facts_tail = np.array([])   
                for rec_item in recommended_ids:
                    rec_facts_head = np.vstack([rec_facts_head,item_facts_head[rec_item]]) if rec_facts_head.size else item_facts_head[rec_item]
                    rec_facts_tail = np.vstack([rec_facts_tail,item_facts_tail[rec_item]]) if rec_facts_tail.size else item_facts_tail[rec_item]
                rec_facts = np.vstack([rec_facts_head, rec_facts_tail])

                # TODO: (rel, node) format for dics?
                # perform critique selection
                crit_node, crit_pair = select_critique(ht_facts, rec_facts, crit_args.critique_mode, pop_counts, items_facts_tail_gt)

                # store most recent crit feeback
                if crit_node != None: 
                    prev_crit = np.vstack([prev_crit, np.array([[crit_pair[0], crit_pair[1]]])])

                # item is using movies related, rather than direct crit from user
                if crit_args.critique_target == 'object':
                    true_object_embedding= get_emb(crit_node, model)

                    # get likes * feedback node 
                    x_true_f = rel_emb[0,0].cpu().numpy() * true_object_embedding[1].cpu().numpy()
                    x_true_inv = rel_emb[0,1].cpu().numpy() * true_object_embedding[0].cpu().numpy()


                # make stack of likes * items related to feedback node
                elif args.critique_target == 'item':
                    liked_items_list = obj2items[crit_node]

                    # get and stack things
                    liked_embeddings_list_f = []
                    liked_embeddings_list_inv = []
                    for x in liked_items_list[:10]:
                        liked_embeddings_list_f.append(get_emb(torch.tensor(x),model)[1])
                        liked_embeddings_list_inv.append(get_emb(torch.tensor(x),model)[0])
                    liked_embeddings_f = torch.stack(liked_embeddings_list_f, dim=0)
                    liked_embeddings_inv = torch.stack(liked_embeddings_list_inv, dim=0)
                    true_object_embedding_f = torch.reshape(liked_embeddings_f,(liked_embeddings_f.shape[0],model_args.emb_dim))
                    true_object_embedding_inv = torch.reshape(liked_embeddings_inv,(liked_embeddings_inv.shape[0],model_args.emb_dim))
                
                    x_true_f = rel_emb[0][0].cpu().numpy() * true_object_embedding_f.cpu().numpy()
                    x_true_inv = rel_emb[0][1].cpu().numpy() * true_object_embedding_inv.cpu().numpy()

                # labels (potential for neg samples), fun!!

                y = np.ones(np.shape(x_true_f)[0])
          
                # performing Laplace approx
                etta = etta_dict[session_no] # ew!!

                # use updated user emb otherwise
                if session_no == 1:
                    user_emb_f = user_emb[0].cpu().numpy()
                    user_emb_inv = user_emb[1].cpu().numpy()
                    user_prec_f = priors.user_prec_f
                    user_prec_inv = priors.user_prec_inv

                # TODO: build gauss bp 
                if crit_args.evidence_type == 'direct':
                    # TODO: update this w prior
                    updater_f = Updater(x_true_f, y, user_emb_f, user_prec_f, model_args, crit_args, device, etta)
                    updater_inv = Updater(x_true_inv, y, user_emb_inv, user_prec_inv, model_args, crit_args, device, etta)
                    try:
                
                        user_emb_f, user_prec_f = updater_f.compute_laplace_approximation()
                        user_emb_inv, user_prec_inv = updater_inv.compute_laplace_approximation()
                        #print("updated")
                    except:
                        pass


                elif crit_args.evidence_type == 'indirect':
                    updater_f = Updater_Indirect(x_true_f, y, user_emb_f, tau_prior_f, tau_z_f, args, device, etta, rel_emb[0][0].cpu().numpy())
                    updater_inv = Updater_Indirect(x_true_inv, y, user_emb_inv, tau_prior_inv, tau_z_inv, args, device, etta, rel_emb[0][1].cpu().numpy())
                    user_emb_f, tau_prior_f = updater_f.compute_laplace_approximation()
                    user_emb_inv, tau_prior_inv = updater_inv.compute_laplace_approximation()


                # update prior class with new user data
                user_emb_updated = (torch.tensor(user_emb_f).to(device), torch.tensor(user_emb_inv).to(device))
                ranked = get_scores(user_emb_updated, rel_emb[0], item_emb, dataloader, model_args.learning_rel)
                rank = get_rank(ranked, [gt], all_gt, id2index)
                rank_track_user[j, session_no] = rank
        rank_track = np.vstack([rank_track, rank_track_user])       
    # save results
    if mode == 'val':
        print(args.test_name)
        sys.exit()
        mrr = save_metrics_critiquing(rank_track, args.test_name, mode)
        return mrr
    else:
        save_metrics_critiquing(rank_track, args.test_name, mode)
    
if __name__ == '__main__':
    args = get_args_critique()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #model_path = os.path.join(args.load_name, 'fold_{}'.format(args.fold), 'models','best_model.pt')
    model_path = os.path.join(args.load_name, 'best_model.pt')
    model = torch.load(model_path).to(device)
    print("model loaded")
    critiquing(model, args, 'test')


    