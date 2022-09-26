from cmath import tau
import os, pickle, sys, argparse, yaml
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from updater import Updater
from dataload import DataLoader
from tester import get_array, get_emb, get_scores, get_rank, GetGT
from recommender import select_critique, remove_chosen_critiques, obj2item
from utils.plots import RankTrack, rank_plot, save_metrics_critiquing

def get_args_critique():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help='name of folder where model is')
    parser.add_argument('-dataset', default='ML_FB', type=str, help='Movielens dataset')
    parser.add_argument('-emb_dim', default=128, type=int, help='embedding dimension')
    parser.add_argument('-alpha', default=0.01, type=float, help='Learning rate for Laplace Approximation')
    #TODO: Have multiple ettas for each session
    #parser.add_argument('-etta', default=1.0, type=float, help='Precision for Laplace Approximation')
    #TODO: This is bad
    parser.add_argument('-ettaone', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-ettatwo', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-ettathree', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-ettafour', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-num_users', default=100, type=int, help='number of users')
    parser.add_argument('-session_length', default=5, type=int, help='number of critiquing sessions')
    parser.add_argument('-update_type', default='gaussian', type=str, help='laplace or gaussian')
    parser.add_argument('-critique_mode', default='random', type=str, help='random or pop or diff')
    parser.add_argument('-critique_target', default='item', type=str, help='object or item')
    parser.add_argument('-likelihood_precision', default=1e-2, type=float, help='likelihood precision')
    parser.add_argument('-tau_prior', default=1e-2, type=float, help='prior precision')
    parser.add_argument('-fold', default=0, type=int, help='fold number')

    args = parser.parse_args()
    return args


def critiquing(model, args, val_or_test):

    etta_dict = {1: args.ettaone, 2: args.ettatwo, 3: args.ettathree, 4: args.ettafour}

    #load required data
    data_path = "datasets/" + args.dataset
    with open(os.path.join(data_path, 'items_facts_head.pkl'), 'rb') as f:
        items_facts_head = pickle.load(f)
    ### This dict contains facts about the item in which the item is the tail of the triple
    with open(os.path.join(data_path, 'items_facts_tail.pkl'), 'rb') as f:
      items_facts_tail = pickle.load(f)
    ### This dict maps objects (non-item entities) to items, e.g., "Bergman" to "Saraband", "Wild Strawberries", etc
    with open(os.path.join(data_path, 'obj2items.pkl'), 'rb') as f:
      obj2items=pickle.load(f)
    ### This dict contains counts for how many times an object was repeated in the KG. This will be used as a measure of its popularity
    with open(os.path.join(data_path, 'pop_counts.pkl'), 'rb') as f:
      pop_counts=pickle.load(f)
        

    # load model and get parameter from file
    path = os.path.join('results', args.test_name, 'fold_{}'.format(args.fold))
    with open(os.path.join(path, 'info.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        for key in yml.keys():
            setattr(args, key, yml[key])
    # print important hps
    print('alpha: {}\tetta: {}\tcritique mode: {}\tcritique target: {}'.format(
        args.alpha, args.etta, args.mode, args.critique_target))
    # load datasets
    print('loading dataset: {}'.format(args.dataset))
    dataloader = DataLoader(args)



    # make arrays with embeddings, and dict to map 
    rec_h, rec_t, id2index, index2id = get_array(model, dataloader, args, rec=True)
    item_emb = (rec_h, rec_t)

    # TODO: setup for tests??
    # load gt info and track rank scores classes

    #why is this "train"?
    #get_gt = GetGT(dataloader.fold, 'train')
    get_gt = GetGT(dataloader.fold, val_or_test)

    # get all relationships
    with torch.no_grad():
        rels = dataloader.num_rel
        r = torch.linspace(0, rels - 1, rels).long().to('cuda')
        rel_f = model.rel_embs(r)
        rel_inv = model.rel_inv_embs(r)
        rel_emb = torch.stack((rel_f, rel_inv))
        rel_emb = torch.permute(rel_emb, (1,0,2))


    # all users, gt selects test / train gts
    data = dataloader.rec_test if val_or_test == 'test' else dataloader.rec_val

    #shouldn't we just enumerate ul_test keys?
    all_users = torch.tensor(data[:,0]).to('cuda')
    #all_users = np.fromiter(get_gt.maps[1].keys(), dtype = 'int')
    rank_track = RankTrack()

    # main test loop
    for i, user in tqdm(enumerate(all_users)):
        # TODO: abstract this process in tester and call one function
        test_emb = get_emb(user, model)
        ranked = get_scores(test_emb, rel_emb[0], item_emb, dataloader, args.learning_rel)
        recommended_items = [index2id[int(x)] for x in ranked[0:20]]
        test_gt, all_gt, train_gt = get_gt.get(user.cpu().item())
        # for calculating R-precision

        if test_gt == None: continue
        pre_rank = get_rank(ranked, test_gt, all_gt, id2index)
        #TODO: Do we need rprec here too?
        rprec = np.array([0])
        rank_track.update(pre_rank, rprec, 0)

        for gt in test_gt:
            
            ### These arrays will keep track of the previous critique, so that we avoid repetition of critiques (one user saying "I want Spielberg" multiple times) 
            previous_critiques=np.array([[0,0]])
            critique_selection_data=np.vstack([items_facts_head[gt],items_facts_tail[gt]])
        
            for session_no in range (1, args.session_length):
            ### double-checking to see if we have any facts remaining about the item and removing previous critiques from the critique selection data 
            
            ### note that we may run out of facts due to consecutive eliminations at critiquing sessions
                      ### double-checking to see if we have any facts remaining about the item and removing previous critiques from the critique selection data 
          ### note that we may run out of facts due to consecutive eliminations at critiquing sessions
                if session_no==1:
                    tau_prior= args.tau_prior * np.eye(args.emb_dim)
                    
                if np.shape(critique_selection_data)[0] == 0:
                    continue

                if np.shape(critique_selection_data)[0]>1:
                    dims = np.maximum(previous_critiques.max(0), critique_selection_data.max(0)) + 1
                    critique_selection_data = critique_selection_data[~np.in1d(np.ravel_multi_index(critique_selection_data.T, dims), np.ravel_multi_index(previous_critiques.T, dims))]
          ### Get the facts related to the recommended items
                
                items_facts_tail_gt = np.hstack([items_facts_tail[gt],np.full((np.shape(items_facts_tail[gt])[0],1),-1)])
                
                
          #select critique from the adjusted data. How do we pick the critique? the "critique_mode" argument decides this. More details are explained in the "recommender.py"
          ###output is the selected critique and the corresponding fact (e.g.,"Bergman", " directed_by, Bergman")
                rec_facts_head = np.array([])   
                rec_facts_tail = np.array([])   
                for rec_item in recommended_items:
                    rec_facts_head = np.vstack([rec_facts_head,items_facts_head[rec_item]]) if rec_facts_head.size else items_facts_head[rec_item]
                    rec_facts_tail = np.vstack([rec_facts_tail,items_facts_tail[rec_item]]) if rec_facts_tail.size else items_facts_tail[rec_item]
                rec_facts=np.vstack([rec_facts_head,rec_facts_tail])
                

          ### Perform critique selection
                object , critique_fact = select_critique(critique_selection_data,rec_facts,args.critique_mode,pop_counts,items_facts_tail_gt)


          #t3=time.time()
          #print(t3-t2)
                if object != None: ###checking whether a critique was selected (maybe we ran out of facts for the ground truth and don't want to perform update anymore)
                    previous_critiques = np.vstack([previous_critiques,np.array([[critique_fact[0],critique_fact[1]]])])
              ### The argument "critique_target" is for deciding on what kind of embedding we want to base our updates on. If it is "object", once the user says: "I like Bergman" we
              ### get the embedding of Bergman multiplied by the "likes relation embedding" as the true example to get the posterior. But if it's "item", we search the kg to get the 
              ### list of movies directed by Bergman and do the update based on that (e.g., user likes Wild Strawberries, user likes Saraband, etc.)

                #Attn: CVXPY doesn't work with torch so we have to convert to numpy (is there any workaround?)
                if args.critique_target == "object":
                    true_object_embedding= get_emb(object, model)

                    X_true_f = np.multiply((rel_emb[0][0].cpu().numpy()),(true_object_embedding[1].cpu().numpy()))
                    X_true_inv = np.multiply((rel_emb[0][1].cpu().numpy()),(true_object_embedding[0].cpu().numpy()))

                elif args.critique_target == "item":
                    liked_items_list = obj2items[object]
                    
                    
                #TODO: this should be a torch array  and X_true should change too
                    #true_object_embedding = torch.empty(size=(10, args.emb_dim))
                    #for i in range(10):
                    #    true_object_embedding[i] = get_emb(liked_items_list[i],model)[1]
                    liked_embeddings_list_f = [get_emb(x,model)[1] for x in liked_items_list[:10]]
                    liked_embeddings_list_inv = [get_emb(x,model)[0] for x in liked_items_list[:10]]
                    #true_object_embedding = torch.FloatTensor(liked_embeddings_list)
                    liked_embeddings_f = torch.stack(liked_embeddings_list_f, dim=0)
                    liked_embeddings_inv = torch.stack(liked_embeddings_list_inv, dim=0)
                    true_object_embedding_f = torch.reshape(liked_embeddings_f,(liked_embeddings_f.shape[0],args.emb_dim))
                    true_object_embedding_inv = torch.reshape(liked_embeddings_inv,(liked_embeddings_inv.shape[0],args.emb_dim))
                    
                    X_true_f = np.multiply((rel_emb[0][0].cpu().numpy()),(true_object_embedding_f.cpu().numpy()))
                    X_true_inv = np.multiply((rel_emb[0][1].cpu().numpy()),(true_object_embedding_inv.cpu().numpy()))


                #X_true = np.reshape(X_true, (args.emb_dim))
 
                y = np.ones(np.shape(X_true)[0])
          #performing Laplace Approximation

                alpha= args.alpha
                etta= etta_dict[session_no]
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                #test_emb_f = np.reshape((test_emb[0].cpu().numpy()),(args.emb_dim))
                test_emb_f = test_emb[0].cpu().numpy()
                test_emb_inv = test_emb[1].cpu().numpy()

                updater_f = Updater(X_true_f, y, test_emb_f, tau_prior, args, device)
                updater_inv = Updater(X_true_inv, y, test_emb_inv, tau_prior, args, device)
                
                #TODO: The reverse direction should also be added (item_h, liked_inv, user_t)
                test_emb_f, tau_prior = updater_f.compute_laplace_approximation()
                test_emb_inv, tau_prior = updater_inv.compute_laplace_approximation()
                
                test_emb_updated = (torch.tensor(test_emb_f).to(device), torch.tensor(test_emb_inv).to(device))
                ranked = get_scores(test_emb_updated, rel_emb[0], item_emb, dataloader, args.learning_rel)
                
                post_rank = get_rank(ranked, [gt], all_gt, id2index)
                rank_track.update(post_rank, rprec, session_no)


    if val_or_test == "val":
        mrr = save_metrics_critiquing(rank_track, args.test_name, val_or_test)
        return mrr
    else:
        save_metrics_critiquing(rank_track, args.test_name, val_or_test)
    


    # critique loop
    # rec class and post. update class
    #for each user:
    #    get embedding 
    #    compute all scores, get top k items to rec 
    #    get test items that user likes gt

    #    get ranks of all gts 
    #    some hp / method where user make a critique (gives info)
    #    update user embedding 

    #    save rank as we update the user emb 


if __name__ == '__main__':
    args = get_args_critique()

    # load model
    #print('loading model from: {}\tfold: {}'.format(args.test_name, args.fold))
    path = os.path.join('results', args.test_name, 'fold_{}'.format(args.fold))
    load_path = os.path.join(path, 'models', 'best_model.pt')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(load_path).to(device)

    critiquing(model, args,"val")


    