from trainer import Trainer
from tester import Tester
from dataload import LoadDataset
import os, pickle, sys
from dataset import Dataset
from measure import Measure
from recommender import Recommender
from updater import Updater
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm
import torch
import numpy as np
import statistics
import wandb
import sys

def get_parameter():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ne', default=0, type=int, help="number of epochs")
    parser.add_argument('-ni', default=0, type=float, help="noise intensity")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="ML_FB", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=1415, type=int, help="batch size")
    parser.add_argument('-save_each', default=50, type=int, help="validate every k epochs")
    parser.add_argument('-max_iters_laplace', default=1000, type=int, help="Maximum number of iterations for Laplace Approximation")
    parser.add_argument('-alpha', default=0.01, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-etta', default=1.0, type=float, help="Precision for Laplace Approximation")
    parser.add_argument('-critique_mode', default="diff", type=str, help="Mode of critiquing")
    parser.add_argument('-critique_target', default="object", type=str, help="Target of User's critique")
    parser.add_argument('-workers', default=8, type=int, help="threads for dataloader")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    loaddataset = LoadDataset("test", args)
    
    # TODO: add support for named checkpoints
    model_path = "models/" + args.dataset + "/" + str(args.ne) + ".chkpnt"
    data_path = "datasets/" + args.dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location='cpu')
    model.eval()

    items_list = loaddataset.rec_items
    users_likes = loaddataset.user_likes_map
    
    # why only 0:2??
    users_list = list(users_likes.keys())[0:2]
    # UPDATE: use this (more general, user_likes isn't required then)
    #users_list = list(loaddataset.users)

    # TODO: where doth this come from??
    ### This dict contains facts about the item in which the item is the head of the triple, e.g., (Saraband, directed_by, Bergman)
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

    items_embeddings_head = np.zeros([len(items_list),args.emb_dim])
    items_embeddings_tail = np.zeros([len(items_list),args.emb_dim])
    items_index = dict(zip(items_list,list(range(0,len(items_list)))))
    items_index_inverse = dict(zip(list(range(0,len(items_list))),items_list))
    
    # UPDATE: this is just len(users_list)
    users_embeddings_head = np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_tail = np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_head_proj = np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_tail_proj = np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])

    # TODO: is this not the same as the beginning of tester.py?

    ### Getting the embedding of each item and storing in an array
    ### The row number that contains embeddings of each item is stored in a dict
    for item in items_list:
        t = torch.tensor([item]).long()
        item_embedding_head = model.ent_h_embs(t)
        item_embedding_tail = model.ent_t_embs(t)
        to_head = (item_embedding_head.cpu().detach().numpy()).reshape((1,args.emb_dim))
        to_tail = (item_embedding_tail.cpu().detach().numpy()).reshape((1,args.emb_dim))  

        index = items_index[item]
        items_embeddings_head[index]=to_head
        items_embeddings_tail[index]=to_tail

    ### Getting the embedding of each user and storing in an array
    for user in users_list:
        h = torch.tensor([user]).long()
        r = torch.tensor([47]).long()

        user_embedding_head = model.ent_h_embs(h)
        user_embedding_tail = model.ent_t_embs(h)
        likes_embedding = model.rel_embs(r).detach().numpy()
        likes_embedding_inv = model.rel_inv_embs(r).detach().numpy()
        to_head = (user_embedding_head.detach().numpy()).reshape((1,args.emb_dim))
        to_tail = (user_embedding_tail.detach().numpy()).reshape((1,args.emb_dim))
        users_embeddings_head[user-users_list[0]] = to_head
        users_embeddings_tail[user-users_list[0]] = to_tail
        users_embeddings_head_proj[user-users_list[0]] = np.multiply((to_head),(likes_embedding))
        users_embeddings_tail_proj[user-users_list[0]] = np.multiply((to_tail),(likes_embedding_inv))
    
    history={}
    
    session_length= 6
    hitatone={0:[],1:[],2:[],3:[],4:[],5:[]}
    hitatthree={0:[],1:[],2:[],3:[],4:[],5:[]}
    hitatfive={0:[],1:[],2:[],3:[],4:[],5:[]}
    hitatten={0:[],1:[],2:[],3:[],4:[],5:[]}
    hitattwenty={0:[],1:[],2:[],3:[],4:[],5:[]}

    #critiquing loop:
    for user_id in tqdm(users_list):
      history[user_id]={}
      user_posterior=users_embeddings_head[user_id-users_list[0]]
      ground_truth=0
      recommender= Recommender(loaddataset,model,user_id,ground_truth,"pre",user_posterior,items_embeddings_head,items_embeddings_tail,users_embeddings_head_proj[user_id-users_list[0]],users_embeddings_tail_proj[user_id-users_list[0]])
      ### Before critiquing begins, get the recommendation and ground truth (gt) rank
      ranked_indices_pre = recommender.pre_critiquing_new()
      recommended_items_pre = [items_index_inverse[k] for k in ranked_indices_pre[0:20]]

      ### each user likes a number of items. Each time, one of the items that the user likes will be deemed as the ground truth
      for ground_truth in users_likes[user_id]:
        user_posterior=users_embeddings_head[user_id-users_list[0]]
        history[user_id][ground_truth]=[]

        ### These arrays will keep track of the previous critique, so that we avoid repetition of critiques (one user saying "I want Spielberg" multiple times) 
        previous_critiques=np.array([[0,0]])
        critique_selection_data=np.vstack([items_facts_head[ground_truth],items_facts_tail[ground_truth]])
        
        ### we make a pre-critiquing recommendation. Output the list of initially recommended items, and the rank of ground truth
        initial_rank= int(np.where(ranked_indices_pre==items_index[ground_truth])[0])
        recommended_items=recommended_items_pre
        history[user_id][ground_truth].append(initial_rank)
        if initial_rank<2:
          hitatone[0].append(1)
        else:
          hitatone[0].append(0)
        if initial_rank<4:
          hitatthree[0].append(1)
        else:
          hitatthree[0].append(0)
        if initial_rank<6:
          hitatfive[0].append(1)
        else:
          hitatfive[0].append(0)
        if initial_rank<11:
          hitatten[0].append(1)
        else:
          hitatten[0].append(0)
        if initial_rank<21:
          hitattwenty[0].append(1)
        else:
          hitattwenty[0].append(0)
        for session_no in range (1,session_length):
          ### double-checking to see if we have any facts remaining about the item and removing previous critiques from the critique selection data 
          ### note that we may run out of facts due to consecutive eliminations at critiquing sessions
          if np.shape(critique_selection_data)[0]>1:
            dims = np.maximum(previous_critiques.max(0), critique_selection_data.max(0)) + 1
            critique_selection_data = critique_selection_data[~np.in1d(np.ravel_multi_index(critique_selection_data.T, dims), np.ravel_multi_index(previous_critiques.T, dims))]
          ### Get the facts related to the recommended items
          items_facts_tail_gt=np.hstack([items_facts_tail[ground_truth],np.full((np.shape(items_facts_tail[ground_truth])[0],1),-1)])

          #select critique from the adjusted data. How do we pick the critique? the "critique_mode" argument decides this. More details are explained in the "recommender.py"
          ###output is the selected critique and the corresponding fact (e.g.,"Bergman", " directed_by, Bergman")
          rec_facts_head = np.array([])   
          rec_facts_tail = np.array([])   
          for rec_item in recommended_items:
            rec_facts_head = np.vstack([rec_facts_head,items_facts_head[rec_item]]) if rec_facts_head.size else items_facts_head[rec_item]
            rec_facts_tail = np.vstack([rec_facts_tail,items_facts_tail[rec_item]]) if rec_facts_tail.size else items_facts_tail[rec_item]
          rec_facts=np.vstack([rec_facts_head,rec_facts_tail])
          ### Perform critique selection
          object , critique_fact = recommender.select_critique(critique_selection_data,rec_facts,args.critique_mode,pop_counts,items_facts_tail_gt)
          if object != None: ###checking whether a critique was selected (maybe we ran out of facts for the ground truth and don't want to perform update anymore)
            previous_critiques=np.vstack([previous_critiques,np.array([[critique_fact[0],critique_fact[1]]])])
              ### The argument "critique_target" is for deciding on what kind of embedding we want to base our updates on. If it is "object", once the user says: "I like Bergman" we
              ### get the embedding of Bergman multiplied by the "likes relation embedding" as the true example to get the posterior. But if it's "item", we search the kg to get the 
              ### list of movies directed by Bergman and do the update based on that (e.g., user likes Wild Strawberries, user likes Saraband, etc.)
            if args.critique_target == "object":
                true_object_embedding=model.ent_t_embs(torch.tensor([object])).long()
                X_true=np.multiply((likes_embedding),(true_object_embedding.detach().numpy()))
            else:
                liked_items_list = obj2items[object]
                X_true = np.multiply((np.take(items_embeddings_tail,liked_items_list,axis=0)),(likes_embedding))

            y = np.ones(np.shape(X_true)[0])
          #performing Laplace Approximation
            max_iters= args.max_iters_laplace
            alpha= args.alpha
            etta= args.etta
            if session_no==1:
              Sigma_prior= 100*np.eye(args.emb_dim)
            updater=Updater(X_true,y,user_posterior,Sigma_prior,user_posterior,args,etta,device)
            mu_prior, _ = updater.SDR_cvxopt(Sigma_prior, X_true, y, user_posterior)
            ###This is the user posterior that will be used as the prior for the next session
            user_posterior=mu_prior
            user_posterior_proj = np.multiply(mu_prior,likes_embedding)
            ###This is the covariance of the posterior
            _,Sigma_prior=updater.compute_laplace_approximation()
            ###Make recommendation again (after Bayesian update)
            recommended_items, rank= recommender.post_critiquing_recommendation(user_posterior_proj,ground_truth)
            history[user_id][ground_truth].append(rank)
            if rank<2:
              hitatone[session_no].append(1)
            else:
              hitatone[session_no].append(0)
            if rank<4:
              hitatthree[session_no].append(1)
            else:
              hitatthree[session_no].append(0)
            if rank<6:
              hitatfive[session_no].append(1)
            else:
              hitatfive[session_no].append(0)
            if rank<11:
              hitatten[session_no].append(1)
            else:
              hitatten[session_no].append(0)
            if rank<21:
              hitattwenty[session_no].append(1)
            else:
              hitattwenty[session_no].append(0)
    
      
      hitatone0=statistics.mean(list(hitatone[0]))
      print("hits@1 session0",hitatone0)
      hitatthree0=statistics.mean(hitatthree[0])
      print("hits@3 session0",hitatthree0)
      hitatfive0=statistics.mean(hitatfive[0])
      print("hits@5 session0",hitatfive0)
      hitatten0=statistics.mean(hitatten[0])
      print("hits@10 session0",hitatten0)
      hitattwenty0=statistics.mean(hitattwenty[0])
      print("hits@20 session0",hitattwenty0)
      hitatone1=(statistics.mean(list(hitatone[1])))
      print("hits@1 session1",hitatone1)
      hitatthree1=(statistics.mean(hitatthree[1]))
      print("hits@3 session1",hitatthree1)
      hitatfive1=(statistics.mean(hitatfive[1]))
      print("hits@5 session1",hitatfive1)
      hitatten1=(statistics.mean(hitatten[1]))
      print("hits@10 session1",hitatten1)
      hitattwenty1=(statistics.mean(hitattwenty[1]))
      print("hits@20 session1",hitattwenty1)
      hitatone2=(statistics.mean(list(hitatone[2])))
      print("hits@1 session2",hitatone2)
      hitatthree2=statistics.mean(hitatthree[2])
      print("hits@3 session2",hitatthree2)
      hitatfive2=statistics.mean(hitatfive[2])
      print("hits@5 session2",hitatfive2)
      hitatten2=statistics.mean(hitatten[2])
      print("hits@10 session2",hitatten2)
      hitattwenty2=statistics.mean(hitattwenty[2])
      print("hits@20 session2",hitattwenty2)
      hitatone3=(statistics.mean(list(hitatone[3])))
      print("hits@1 session3",hitatone2)
      hitatthree3=statistics.mean(hitatthree[3])
      print("hits@3 session3",hitatthree3)
      hitatfive3=statistics.mean(hitatfive[3])
      print("hits@5 session3",hitatfive3)
      hitatten3=statistics.mean(hitatten[3])
      print("hits@10 session3",hitatten3)
      hitattwenty3=statistics.mean(hitattwenty[3])
      print("hits@20 session3",hitattwenty3)
      hitatone4=(statistics.mean(list(hitatone[4])))
      print("hits@1 session4",hitatone4)
      hitatthree4=statistics.mean(hitatthree[4])
      print("hits@3 session4",hitatthree4)
      hitatfive4=statistics.mean(hitatfive[4])
      print("hits@5 session4",hitatfive4)
      hitatten4=statistics.mean(hitatten[4])
      print("hits@10 session4",hitatten4)
      hitattwenty4=statistics.mean(hitattwenty[4])
      print("hits@20 session4",hitattwenty4)
      hitatone5=(statistics.mean(list(hitatone[5])))
      print("hits@1 session5",hitatone5)
      hitatthree5=statistics.mean(hitatthree[5])
      print("hits@3 session5",hitatthree5)
      hitatfive5=statistics.mean(hitatfive[5])
      print("hits@5 session5",hitatfive5)
      hitatten5=statistics.mean(hitatten[5])
      print("hits@10 session5",hitatten5)
      hitattwenty5=statistics.mean(hitattwenty[5])
      print("hits@20 session5",hitattwenty5)

    with open('saved_history.pkl', 'wb') as f:
      pickle.dump(history, f)

