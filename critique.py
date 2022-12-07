import os, sys, pickle, torch, argparse, yaml, time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace

from launch import get_model_args
from updater import Updater, beta_update, beta_update_indirect
from dataload import DataLoader
from tester import get_array, get_emb, get_scores, get_rank, GetGT
from recommender import crit_selector, test_crit
from utils.plots import RankTrack, rank_plot
from utils.updateinfo import UpdateInfo
from utils.crit_utils import InfoTrack, fact_stack, rec_fact_stack, get_d, fake_d, get_dics, sim_selector

def get_args_critique():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_name', default='results/tuned/tilt_small/fold_0/train/train_21', type=str, help='name of folder where model is')
    parser.add_argument('-test_name', default=None, type=str, help='name of folder where model is')

    parser.add_argument('-user_prec', default=1.0, type=float, help='prior cov')
    parser.add_argument('-default_prec', default=1.0, type=float, help='likelihood precision')
    parser.add_argument('-z_prec', default=2.0, type=float, help='item distribution precision indirect case')

    parser.add_argument('-etta', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-alpha', default=0.05, type=float, help='Learning rate for GD in Laplace Approximation')
    parser.add_argument('-multi_k', default=10, type=int, help='number of samples for multi type update')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-num_users', default=1000, type=int, help='number of users')
    parser.add_argument('-sim_k', default=0, type=int, help='number closest movies for direct single testing')

    # single vs mult
    parser.add_argument('-critique_target', default='multi', type=str, help='single or multi')
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    
    # likelihood
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gauss')
    parser.add_argument('-crit_mode', default='diff', type=str, help='random or pop or diff')
    parser.add_argument('-map_finder', default='cvx', type= str, help='cvx or gd')

    # redundant args because of inner_cv
    parser.add_argument('-cluster_check', default=False, type=str, help='run fast version of code')
    parser.add_argument('-cv_tune_name', default='tuned', type=str, help='upper level folder name')
    parser.add_argument('-samples', default=10000, type=int, help='no of samples in tuning')
    parser.add_argument('-batch', default=4, type=int, help='no of simultaneous calls of script')
    parser.add_argument('-folds', default=5, type=int, help='no of folds')
    parser.add_argument('-epochs_all', default=120, type=int, help='no of total epochs')
    parser.add_argument('-tune_type', default='two_stage', type=str, help='two_stage or joint')
    parser.add_argument('-name', default='diff', type=str, help='name of current test')

    args = parser.parse_args()
    return args

def crit_arg_asserts(args): pass

def calc_score(user, d):
    s_for = torch.sum(user[0] * d[0])
    s_back = torch.sum(user[1] * d[1])
    return 0.5*(s_for + s_back)

# main loop
def critiquing(crit_args, mode):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args = Namespace()
    etta = 5 * [crit_args.etta]
    alpha = crit_args.alpha

    # for cluster check
    if crit_args.cluster_check: crit_args.num_users = 10

    # load model and get parameter from file
    if crit_args.test_name is None:
        crit_args.test_name = crit_args.load_name.replace('/train/', '/crit/')
    save_path = os.path.join(crit_args.test_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(crit_args.load_name, 'info.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        for key in yml.keys():
            if key != "test_name":
                setattr(model_args, key, yml[key])

    # TODO: save these in yaml file
    model_args.learning_rel = 'learn'
    model_args.type_checking = 'yes'
    print(crit_args.cluster_check)
    sys.exit()
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
    all_users = np.unique(torch.tensor(data[:, 0]).cpu().numpy(), return_counts=False)

    # for tracking all info (score, distance, rank)
    info_track = InfoTrack(crit_args.session_length)

    # main test loop (each user)
##############################################################
    t0 = time.time()
    rec_k = 10 # TODO: (number of recommended items to user) this must be an hp
    np.random.shuffle(all_users)
    for i, user in enumerate(all_users):
        if i > crit_args.num_users: break

        # get ids of top k recs, and all gt from user
        user_emb = get_emb(user, model, device)
        val_or_test_gt, all_gt, train_gt = get_gt.get(user)

        # iterature through all gt for single user
        for j, gt in enumerate(val_or_test_gt):
            # get all triplets w gt
            gt_ind = id2index[gt]
            gt_facts = fact_stack(item_facts_head[gt], item_facts_tail[gt])

            # get initial rank
            (scores, ranked) = get_scores(user_emb, rel_emb[0], item_emb, model_args.learning_rel)
            rank = get_rank(ranked, [gt], all_gt, id2index) 
            
            # save info
            info_track.store(0, rank=rank+1, score=scores[gt_ind])

            # a few sessions for each user 
            for sn in range(crit_args.session_length):
##############################################################
                if sn == 0: update_info = UpdateInfo(user_emb, etta, crit_args, model_args, device, likes_emb=likes_rel)

                # stack facts, either [-1, rel, tail] or [head, rel, -1]
                rec_ids = [index2id[int(x)] for x in ranked[:rec_k]]
                rec_facts = rec_fact_stack(rec_ids, item_facts_head, item_facts_tail)
                if gt_facts.shape[0] <= 0: continue

                #crit = (gt, 0)
                if crit_args.sim_k > 0:
                    # get most item with most similar embedding
                    crit = sim_selector(gt, item_emb, id2index, index2id, device, k=5)
                else:
                    # actual critique selection for real experiments
                    crit = crit_selector(gt_facts, rec_facts, crit_args.crit_mode, pop_counts)

                # get d for p(user | d) bayesian update
                #d, r = fake_d(gt, get_emb, rel_emb[0], model, device, sigma=1.5)
                d = get_d(model, crit, rel_emb, obj2items, get_emb, crit_args, model_args, device)
                update_info.store(d=d, crit_rel_emb=rel_emb[crit[1]])

                # perform update
                if crit_args.evidence_type == 'direct':
                    beta_update(update_info, sn, crit_args, model_args, device, crit_args.update_type, crit_args.map_finder, etta, alpha)
                if crit_args.evidence_type == 'indirect':
                    beta_update_indirect(update_info, sn, crit_args, model_args, device, crit_args.update_type, crit_args.map_finder, etta, alpha)

                # track rank in training
                new_user_emb, _ = update_info.get_priorinfo()
                (scores, ranked) = get_scores(new_user_emb, rel_emb[0], item_emb, model_args.learning_rel)
                post_rank = get_rank(ranked, [gt], all_gt, id2index)

                # save info
                info_track.store(sn+1, rank=post_rank+1, score=scores[gt_ind], dist=(new_user_emb, d))

    # save results
    info_track.save(crit_args.test_name)

if __name__ == '__main__':
    crit_args = get_args_critique()
    critiquing(crit_args, 'val')
