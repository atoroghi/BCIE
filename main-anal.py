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
import time
import scipy.stats as st
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_run', default='64.222', type=str, help="number of run for model load")
    parser.add_argument('-ne', default=1000, type=int, help="number of epochs")
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
    parser.add_argument('-num_users', default=100, type=int, help="number of users")
    parser.add_argument('-initial_Sigma', default=1e-2, type=float, help="initial prior precision")
    parser.add_argument('-neg_power', default=0, type=float, help="power for neg sampling disribution")
    parser.add_argument('-etta_cvx', default=0, type=float, help="etta for updading cvx problem")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    #dataset = Dataset(args.dataset)
    loaddataset=LoadDataset('test', args)
    model_path="results/" + str(args.num_run)+"/models/epoch "+str(args.ne) + ".chkpnt"
    #model_path="models/" + "ML" + "/" + "10000" + ".chkpnt"
    data_path="datasets/" + args.dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location='cpu')
    model.eval()

   # wandb.login(key="d606ae06659873794e6a1a5fb4f55ffc72aac5a1")
   # wandb.init(project="critiquing",config={"lr": 0.1})
   # os.environ['WANDB_API_KEY']='d606ae06659873794e6a1a5fb4f55ffc72aac5a1'
   # os.environ['WANDB_USERNAME']='atoroghi'
  #  wandb.config.update(args)
    #wandb.init(project="critiquing",config={"lr": 0.1})
    #wandb.config.update(args,allow_val_change=True)
    items_list= loaddataset.rec_items
    num_items=len(items_list)
    users_likes= loaddataset.user_likes_map
    #users_list=users_likes.keys()
    users_list=list(users_likes.keys())[-args.num_users:]
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

    with open(os.path.join(data_path, 'critiques_analogical.pkl'), 'rb') as f:
        critiques_analogical=pickle.load(f)

    items_embeddings_head=np.zeros([len(items_list),args.emb_dim])
    items_embeddings_tail=np.zeros([len(items_list),args.emb_dim])
    items_index=dict(zip(items_list,list(range(0,len(items_list)))))
    items_index_inverse=dict(zip(list(range(0,len(items_list))),items_list))
    users_embeddings_head=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_tail=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_head_proj=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_tail_proj=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    ### Getting the embedding of each item and storing in an array
    ### The row number that contains embeddings of each item is stored in a dict
    for item in items_list:
        t = torch.tensor([item]).long()
        item_embedding_head = model.ent_h_embs(t)
        item_embedding_tail = model.ent_t_embs(t)
        to_head=(item_embedding_head.cpu().detach().numpy()).reshape((1,args.emb_dim))
        to_tail=(item_embedding_tail.cpu().detach().numpy()).reshape((1,args.emb_dim))  
        index=items_index[item]
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
        to_head=(user_embedding_head.detach().numpy()).reshape((1,args.emb_dim))
        to_tail=(user_embedding_tail.detach().numpy()).reshape((1,args.emb_dim))
        users_embeddings_head[user-users_list[0]]=to_head
        users_embeddings_tail[user-users_list[0]]=to_tail
        users_embeddings_head_proj[user-users_list[0]]=np.multiply((to_head),(likes_embedding))
        users_embeddings_tail_proj[user-users_list[0]]=np.multiply((to_tail),(likes_embedding_inv))

    
    history={}
    
    session_length=2
    hitatone={0:[],1:[]}
    hitatthree={0:[],1:[]}
    hitatfive={0:[],1:[]}
    hitatten={0:[],1:[]}
    hitattwenty={0:[],1:[]}

    MAR_pre=[]
    MAR_post=[]
    MAR_post_all={1:[]}



    #critiquing loop:
    for user_id in tqdm(users_list):
        history[user_id]={}
        for ground_truth_object in critiques_analogical.keys():
            user_posterior=users_embeddings_head[user_id-users_list[0]]
            ground_truth=critiques_analogical[ground_truth_object]
            recommender= Recommender(loaddataset,model,user_id,ground_truth,"pre",user_posterior,items_embeddings_head,items_embeddings_tail,users_embeddings_head_proj[user_id-users_list[0]],users_embeddings_tail_proj[user_id-users_list[0]])
            ranked_indices_pre = recommender.pre_critiquing_new()
            initial_rank = int(np.where(ranked_indices_pre==items_index[ground_truth])[0])
            history[user_id][ground_truth]=[]
            mar=(num_items-initial_rank)/(num_items-1)
            MAR_pre.append(mar)
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
           ### Perform critique selection
            object = ground_truth_object

            if object != None: ###checking whether a critique was selected (maybe we ran out of facts for the ground truth and don't want to perform update anymore)
           
                if args.critique_target == "object":
                    true_object_embedding=model.ent_t_embs(torch.tensor([object])).long()
                #X_true=np.vstack([X_true,np.multiply((likes_embedding),(true_object_embedding.detach().numpy()))]) if X_true.size else np.multiply((likes_embedding),(true_object_embedding.detach().numpy()))
                    X_true=np.multiply((likes_embedding),(true_object_embedding.detach().numpy()))
                else:
                    liked_items_list = obj2items[object]
                    liked_indices_list = [items_index[x] for x in liked_items_list[0:10]]
                #X_true = np.vstack([X_true,np.multiply((np.take(items_embeddings_tail,liked_indices_list,axis=0)),(likes_embedding))]) if X_true.size else np.multiply((np.take(items_embeddings_tail,liked_indices_list,axis=0)),(likes_embedding))
                    X_true = np.multiply((np.take(items_embeddings_tail,liked_indices_list,axis=0)),(likes_embedding))
                y = np.ones(np.shape(X_true)[0])
          #performing Laplace Approximation
                max_iters= args.max_iters_laplace
                alpha= args.alpha
                etta= args.etta
                session_no = 1
                Sigma_prior= args.initial_Sigma*np.eye(args.emb_dim)
                updater=Updater(X_true,y,user_posterior,Sigma_prior,user_posterior,args,etta,device,args.etta_cvx)
            ##mu_prior, _ = updater.SDR_cvxopt(Sigma_prior, X_true, y, user_posterior)
  
            ###This is the user posterior that will be used as the prior for the next session
            ##user_posterior=mu_prior
            ##user_posterior_proj = np.multiply(mu_prior,likes_embedding)
            ###This is the covariance of the posterior
            ##_,Sigma_prior=updater.compute_laplace_approximation()
                mu_prior,Sigma_out = updater.compute_laplace_approximation()
                user_posterior=mu_prior
                Sigma_prior=Sigma_out
                user_posterior_proj = np.multiply(mu_prior,likes_embedding)

            ###Make recommendation again (after Bayesian update)
                recommended_items, rank= recommender.post_critiquing_recommendation(user_posterior_proj,ground_truth)

                mar_post=(num_items-rank)/(num_items-1)
                MAR_post.append(mar_post)
                #history[user_id][ground_truth].append(rank)
                MAR_post_all[session_no].append(mar_post)
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
    
    mean_MAR_pre=statistics.mean(MAR_pre)
    print("MAR pre:",mean_MAR_pre)
    print("CI OF MAR pre:",1.96*(np.std(MAR_pre))/np.sqrt(1000))
    mean_MAR_post=statistics.mean(MAR_post)
    print("MAR post:",mean_MAR_post)
    mean_MAR_post1=statistics.mean(MAR_post_all[1])
    print("MAR post 1:",mean_MAR_post1)
    print("CI OF MAR post 1:",1.96*(np.std(MAR_post_all[1]))/np.sqrt(1000))
    hitatone0=statistics.mean(list(hitatone[0]))
    print("hits@1 session0",hitatone0)
    print("CI of hits@1 session0",1.96*(np.std(list(hitatone[0])))/np.sqrt(1000))
    hitatthree0=statistics.mean(hitatthree[0])
    print("hits@3 session0",hitatthree0)
    print("CI of hits@3 session0",1.96*(np.std(list(hitatthree[0])))/np.sqrt(1000))
    hitatfive0=statistics.mean(hitatfive[0])
    print("hits@5 session0",hitatfive0)
    print("CI of hits@5 session0",1.96*(np.std(list(hitatfive[0])))/np.sqrt(1000))
    hitatten0=statistics.mean(hitatten[0])
    print("hits@10 session0",hitatten0)
    print("CI of hits@1 session0",1.96*(np.std(list(hitatten[0])))/np.sqrt(1000))
    hitattwenty0=statistics.mean(hitattwenty[0])
    print("hits@20 session0",hitattwenty0)
    print("CI of hits@1 session0",1.96*(np.std(list(hitattwenty[0])))/np.sqrt(1000))
    hitatone1=(statistics.mean(list(hitatone[1])))
    print("hits@1 session1",hitatone1)
    print("CI of hits@1 session1",1.96*(np.std(list(hitatone[1])))/np.sqrt(1000))
    hitatthree1=(statistics.mean(hitatthree[1]))
    print("hits@3 session1",hitatthree1)
    print("CI of hits@3 session1",1.96*(np.std(list(hitatthree[1])))/np.sqrt(1000))
    hitatfive1=(statistics.mean(hitatfive[1]))
    print("hits@5 session1",hitatfive1)
    print("CI of hits@5 session1",1.96*(np.std(list(hitatfive[1])))/np.sqrt(1000))
    hitatten1=(statistics.mean(hitatten[1]))
    print("hits@10 session1",hitatten1)
    print("CI of hits@10 session1",1.96*(np.std(list(hitatten[1])))/np.sqrt(1000))
    hitattwenty1=(statistics.mean(hitattwenty[1]))
    print("hits@20 session1",hitattwenty1)
    print("CI of hits@20 session1",1.96*(np.std(list(hitattwenty[1])))/np.sqrt(1000))

#    wandb.log({
#        "hits@1 session0":hitatone0,"hits@3 session0":hitatthree0,"hits@5 session0":hitatfive0,"hits@10 session0":hitatten0,"hits@20 session0":hitattwenty0,
#"hits@1 session1":hitatone1,"hits@3 session1":hitatthree1,"hits@5 session1":hitatfive1,"hits@10 session1":hitatten1,"hits@20 session1":hitattwenty1,
#"hits@1 session2":hitatone2,"hits@3 session2":hitatthree2,"hits@5 session2":hitatfive2,"hits@10 session2":hitatten2,"hits@20 session2":hitattwenty2,
#"hits@1 session3":hitatone3,"hits@3 session3":hitatthree3,"hits@5 session3":hitatfive3,"hits@10 session3":hitatten3,"hits@20 session3":hitattwenty3,
#"hits@1 session4":hitatone4,"hits@3 session4":hitatthree4,"hits@5 session4":hitatfive4,"hits@10 session4":hitatten4,"hits@20 session4":hitattwenty4,
#"hits@1 session5":hitatone5,"hits@3 session5":hitatthree5,"hits@5 session5":hitatfive5,"hits@10 session5":hitatten5,"hits@20 session5":hitattwenty5
#      })

          

#    with open('saved_history.pkl', 'wb') as f:
#      pickle.dump(history, f)

