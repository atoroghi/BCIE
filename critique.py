import os, sys, pickle, torch, argparse, yaml, time, re
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
from utils.crit_utils import InfoTrack, fact_stack, rec_fact_stack, get_d, fake_d, get_dics, sim_selector, get_postdiff

def get_args_critique():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_name', default='results/tuned/tilt_small/fold_0/train/train_21', type=str, help='name of folder where model is')
    parser.add_argument('-test_name', default=None, type=str, help='name of folder where model is')
    parser.add_argument('-objective', default='hits', type=str, help='hits or rank')

    parser.add_argument('-user_prec', default=42123.409928333158, type=float, help='prior cov')
    parser.add_argument('-default_prec', default=11292.058464079433, type=float, help='likelihood precision')
    #parser.add_argument('-user_prec', default=0.010857149510475995, type=float, help='prior cov')
    #parser.add_argument('-default_prec', default=1.0042732995119225e-05, type=float, help='likelihood precision')
    parser.add_argument('-z_prec', default=2.0, type=float, help='item distribution precision indirect case')
    parser.add_argument('-z_mean', default=0.0, type=float, help='item distribution mean indirect case')

    parser.add_argument('-etta', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-alpha', default=0.01, type=float, help='Learning rate for GD in Laplace Approximation')
    parser.add_argument('-multi_k', default=10, type=int, help='number of samples for multi type update')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-num_users', default=1000, type=int, help='number of users')
    parser.add_argument('-sim_k', default=0, type=int, help='number closest movies for direct single testing')

    # single vs mult
    parser.add_argument('-critique_target', default='multi', type=str, help='single or multi')
    parser.add_argument('-evidence_type', default='direct', type=str, help='direct or indirect')
    parser.add_argument('-no_hps', default=4, type=int, help='number of considered hps for tuning')
    
    # likelihood
    parser.add_argument('-update_type', default='gauss', type=str, help='laplace or gauss')
    parser.add_argument('-crit_mode', default='diff', type=str, help='random or pop or diff')
    parser.add_argument('-map_finder', default='cvx', type= str, help='cvx or gd')

    # redundant args because of inner_cv
    parser.add_argument('-cluster_check', default='False', type=str, help='run fast version of code')
    parser.add_argument('-cv_tune_name', default='tuned', type=str, help='upper level folder name')
    parser.add_argument('-samples', default=10000, type=int, help='no of samples in tuning')
    parser.add_argument('-batch', default=4, type=int, help='no of simultaneous calls of script')
    parser.add_argument('-folds', default=5, type=int, help='no of folds')
    parser.add_argument('-epochs_all', default=120, type=int, help='no of total epochs')
    parser.add_argument('-tune_type', default='two_stage', type=str, help='two_stage or joint')
    parser.add_argument('-param_tuning', default='per_session', type=str, help='per_session or together')
    parser.add_argument('-name', default='diff', type=str, help='name of current test')
    parser.add_argument('-fold', default=0, type=int, help='fold')
    parser.add_argument('-session', default=0, type=int, help='session used for per_session param_tuning')
    

    args = parser.parse_args()
    return args

def crit_arg_asserts(args): pass

def calc_score(user, d):
    s_for = torch.sum(user[0] * d[0])
    s_back = torch.sum(user[1] * d[1])
    return 0.5*(s_for + s_back)
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
def best_model(path):
    folders = os.listdir(path)
    folders = [f for f in folders if 'train' in f]
    folders = sorted(folders, key=natural_key)

    # get performance for each model in a fold
    perf, arg_perf = [], []

    for f in folders:
        try:
            scores = np.load(os.path.join(path, f, 'stop_metric.npy'), allow_pickle=True)
            perf.append(np.max(scores))
            arg_perf.append(np.argmax(scores))
        except:
            print('skipped: ', f)

    best_run = np.argmax(perf)
    best_score = np.max(perf)
    best_epoch = arg_perf[np.argmax(perf)]
    # best_folder is not necessarily best_run
    return (best_score, best_run, best_epoch, folders[best_run])

# main loop
def critiquing(crit_args, mode):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args = Namespace()
    if mode == 'val':
        ettas = crit_args.session_length * [crit_args.etta]
        alpha = crit_args.alpha
        user_precs = crit_args.session_length * [crit_args.user_prec]
        default_precs = crit_args.session_length * [crit_args.default_prec]
        z_means = crit_args.session_length * [crit_args.z_mean]
        z_precs = crit_args.session_length * [crit_args.z_prec]
    elif mode == 'test':
        # in the case a list is passed in for hps in different sessions
        try:
            ettas = crit_args.ettas; alpha = crit_args.alpha; user_precs = crit_args.user_precs; default_precs = crit_args.default_precs
            z_means = crit_args.z_means; z_precs = crit_args.z_precs
        # otherwise
        except:
            ettas = crit_args.session_length * [crit_args.etta]; alpha = crit_args.alpha; user_precs = crit_args.session_length * [crit_args.user_prec]
            default_precs = crit_args.session_length * [crit_args.default_prec]; z_means = crit_args.session_length * [crit_args.z_mean]; z_precs = crit_args.session_length * [crit_args.z_prec]


    # for cluster check
    #if crit_args.cluster_check: crit_args.num_users = 10
    if crit_args.cluster_check == 'True':
        crit_args.num_users = 10

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

    if crit_args.param_tuning == 'per_session' and mode == 'val':
        # reading hps for best models of previous sessions 
        if crit_args.session > 0:
            prev_path = os.path.join(crit_args.load_name,'../..',crit_args.name, 'session_{}'.format(crit_args.session-1))
            (_, _, _, best_folder_prev) = best_model(prev_path)
            args_path = os.path.join(prev_path, best_folder_prev, 'crit hps.yml')
            with open(args_path, 'r') as f:
                yml = yaml.safe_load(f)
            user_precs = yml['user_precs']
            default_precs[:crit_args.session] = yml['default_precs'][:crit_args.session]
            z_precs[:crit_args.session] = yml['z_precs'][:crit_args.session]
            z_means[:crit_args.session] = yml['z_means'][:crit_args.session]
            ettas[:crit_args.session] = yml['ettas'][:crit_args.session]



    # save current session hps
    save_dict = {}
    for k, v in vars(crit_args).items():
        save_dict.update({k : str(v)})
    save_dict['user_precs'] = user_precs; save_dict['default_precs'] = default_precs; save_dict['z_precs'] = z_precs
    save_dict['z_means'] = z_means; save_dict['ettas'] = ettas


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
    data = dataloader.rec_test if mode == 'test' else dataloader.rec_val[:2000]
    all_users = np.unique(torch.tensor(data[:, 0]).cpu().numpy(), return_counts=False)

    # for tracking all info (score, distance, rank)
    info_track = InfoTrack(crit_args.session_length, crit_args.objective, crit_args.param_tuning, crit_args.session)

    if crit_args.param_tuning == 'per_session':
        last_session = crit_args.session + 1
    elif crit_args.param_tuning == 'together':
        last_session = crit_args.session_length


    # main test loop (each user)
##############################################################
    t0 = time.time()
    rec_k = 10 # TODO: (number of recommended items to user) this must be an hp
    #np.random.shuffle(all_users)
    for i, user in enumerate(all_users):
        #print("user")
        #print(i)

        if i > crit_args.num_users: break

        # get ids of top k recs, and all gt from user

        user_emb = get_emb(user, model, device)
        val_or_test_gt, all_gt, train_gt = get_gt.get(user)
        

        # iterature through all gt for single user
        for j, gt in enumerate(val_or_test_gt):
            #if j >1 : 
                #sys.exit()
            #print("gt:", gt)

            # get all triplets w gt
            gt_ind = id2index[gt]
            gt_facts = fact_stack(item_facts_head[gt], item_facts_tail[gt])

            # get initial rank 
            (scores, ranked) = get_scores(user_emb, rel_emb[0], item_emb, model_args.learning_rel)
            rank = get_rank(ranked, [gt], all_gt, id2index) 


            #print("initial rank:", rank)
            #pre critiquing ranked used for postcritiquingdiff calcualtion
            ranked_pre = 1*ranked
            


            # save info
            info_track.store(0, rank=rank+1, score=scores[gt_ind])


            # a few sessions for each user 
            #for sn in range(crit_args.session_length):
            for sn in range(last_session):
##############################################################
                if sn == 0: 
                    update_info = UpdateInfo(user_emb, ettas, user_precs, default_precs, z_means, z_precs, crit_args, model_args, device, likes_emb=likes_rel)
                else:
                    z_mean = z_means[sn] * torch.ones(model_args.emb_dim).to(device)
                    z_prec = z_precs[sn] * torch.eye(model_args.emb_dim).to(device)
                    likelihood_prec = default_precs[sn] * torch.eye(model_args.emb_dim).to(device)
                    update_info.store(z_mean=z_mean, z_prec=z_prec, likelihood_prec=likelihood_prec)


                # stack facts, either [-1, rel, tail] or [head, rel, -1]
                rec_candidates = ranked[:(rec_k + len(train_gt))]

                rec_candidate_ids = [index2id[int(x)] for x in rec_candidates]
                rec_ids = [x for x in rec_candidate_ids if x not in train_gt][:rec_k]
                #print("recommended items are:")
                #print(rec_ids)
                # we should make sure not to recommend items in train set!
                #rec_ids = [index2id[int(x)] for x in ranked[:rec_k]]
                rec_facts = rec_fact_stack(rec_ids, item_facts_head, item_facts_tail)
                if gt_facts.shape[0] <= 1: continue

                #crit = (gt, 0)
                if crit_args.sim_k > 0:
                    # get most item with most similar embedding
                    crit = sim_selector(gt, item_emb, id2index, index2id, device, k=5)
                else:
                    # actual critique selection for real experiments
                    try:
                        crit, crit_triple = crit_selector(gt_facts, rec_facts, crit_args.crit_mode, pop_counts)
                    except:
                        #print("failure was heppening here")
                        #print("gt_facts")
                        #print(gt_facts)
                        #print("rec_facts")
                        #print(rec_facts)
                        continue

                (crit_node, crit_rel) = crit

                #print("selected critique:")

               #removing the selected critique from gt_facts
                gt_facts = np.delete(gt_facts, np.where(np.all(gt_facts == crit_triple, axis=1))[0][0], axis=0)

                # get d for p(user | d) bayesian update
                #d, r = fake_d(gt, get_emb, rel_emb[0], model, device, sigma=1.5)
                d = get_d(model, crit, rel_emb, obj2items, get_emb, crit_args, model_args, device)
                update_info.store(d=d, crit_rel_emb=rel_emb[crit[1]])


                # perform update
                if crit_args.evidence_type == 'direct':
                    beta_update(update_info, sn, crit_args, model_args, device, crit_args.update_type, crit_args.map_finder, ettas, alpha)
                if crit_args.evidence_type == 'indirect':
                    beta_update_indirect(update_info, sn, crit_args, model_args, device, crit_args.update_type, crit_args.map_finder, ettas, alpha)

                # track rank in training
                new_user_emb, _ = update_info.get_priorinfo()
                #print(new_user_emb[0])
                (scores, ranked) = get_scores(new_user_emb, rel_emb[0], item_emb, model_args.learning_rel)
                post_rank = get_rank(ranked, [gt], all_gt, id2index)
                ranked_post = 1*ranked
                pcd = 0

                # only for single step critiquing experiment
                if crit_args.objective == 'pcd':

                    pcd = get_postdiff(ranked_pre, ranked_post, crit_node, obj2items, all_gt, id2index)

                #print("rank after update")
                #print(post_rank)

                # save info
                info_track.store(sn+1, rank=post_rank+1, score=scores[gt_ind], dist=(new_user_emb, d), pcd=pcd)

            #rec_candidates = ranked[:(rec_k + len(train_gt))]
            #rec_candidate_ids = [index2id[int(x)] for x in rec_candidates]
            #rec_ids = [x for x in rec_candidate_ids if x not in train_gt][:rec_k]
            #print("recommended items are:")
            #print(rec_ids)

    # save results
    info_track.save(crit_args.test_name)

if __name__ == '__main__':
    crit_args = get_args_critique()
    critiquing(crit_args, 'val')
    #critiquing(crit_args, 'test')
