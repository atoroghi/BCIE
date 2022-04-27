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
def get_parameter():
    parser = argparse.ArgumentParser()
    
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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    #dataset = Dataset(args.dataset)
    loaddataset=LoadDataset(args.dataset,"test",10,0)
    model_path="models/" + args.dataset + "/" + str(args.ne) + ".chkpnt"
    #model_path="models/" + "ML" + "/" + "10000" + ".chkpnt"
    data_path="datasets/" + args.dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    #users_list= dataset.users
    #items_list= dataset.items
    #users_likes= dataset.users_likes
    wandb.login(key="d606ae06659873794e6a1a5fb4f55ffc72aac5a1")
    wandb.init(project="critiquing",config={"lr": 0.1})
    os.environ['WANDB_API_KEY']='d606ae06659873794e6a1a5fb4f55ffc72aac5a1'
    os.environ['WANDB_USERNAME']='atoroghi'
    wandb.config.update(args)
    items_list= loaddataset.rec_items
    users_likes= loaddataset.user_likes_map
    #users_list=users_likes.keys()
    users_list=list(users_likes.keys())[0:2]
    items_embeddings_head=(-100)*np.ones([np.max(items_list)+1,args.emb_dim])
    items_embeddings_tail=(-100)*np.ones([np.max(items_list)+1,args.emb_dim])
    users_embeddings_head=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_tail=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_head_proj=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    users_embeddings_tail_proj=np.zeros([np.max(users_list)-users_list[0]+1,args.emb_dim])
    for item in items_list:
        h = torch.tensor([0]).long().to(device)
        r = torch.tensor([0]).long().to(device)
        t = torch.tensor([item]).long().to(device)
        _, _, _, item_embedding_tail, _, _, item_embedding_head = model(h, r, t)
        to_head=(item_embedding_head.detach().numpy()).reshape((1,args.emb_dim))
        to_tail=(item_embedding_tail.detach().numpy()).reshape((1,args.emb_dim))  
        items_embeddings_head[item]=to_head
        items_embeddings_tail[item]=to_tail
    for user in users_list:
        h = torch.tensor([user]).long().to(device)
        r = torch.tensor([47]).long().to(device)
        t = torch.tensor([0]).long().to(device)
        _, user_embedding_head, likes_embedding, _, user_embedding_tail, likes_embedding_inv, _ = model(h, r, t)
        to_head=(user_embedding_head.detach().numpy()).reshape((1,args.emb_dim))
        to_tail=(user_embedding_tail.detach().numpy()).reshape((1,args.emb_dim))
        users_embeddings_head[user-users_list[0]]=to_head
        users_embeddings_tail[user-users_list[0]]=to_tail
        likes_embedding_proj=(likes_embedding.detach().numpy())
        likes_embedding_inv_proj=(likes_embedding_inv.detach().numpy())
        users_embeddings_head_proj[user-users_list[0]]=np.multiply((to_head),(likes_embedding_proj))
        users_embeddings_tail_proj[user-users_list[0]]=np.multiply((to_tail),(likes_embedding_inv_proj))

    kg_path = "datasets/" + "ML_FB"
    kg_data = np.load(os.path.join(kg_path, 'kg.npy'), allow_pickle=True)
    likes_embedding=likes_embedding.detach().numpy()

    
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
      for ground_truth in users_likes[user_id]:
        history[user_id][ground_truth]=[]
        previous_critiques_head=np.array([[0,0,0]])
        previous_critiques_tail = np.array([[0,0,0]])
        critique_selection_data=kg_data
        # pre-critiquing recommendation. Output the list of initially recommended items, and the rank of ground truth
        recommender= Recommender(loaddataset,model,user_id,ground_truth,"pre",user_posterior,items_embeddings_head,items_embeddings_tail,users_embeddings_head_proj[user_id-users_list[0]],users_embeddings_tail_proj[user_id-users_list[0]])
        recommended_items, initial_rank= recommender.pre_critiquing_recommendation()
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
          previous_critiques=np.vstack([previous_critiques_head,previous_critiques_tail])
          #removing previous critiques from the critique selection data
          dims = np.maximum(previous_critiques.max(0), critique_selection_data.max(0)) + 1
          critique_selection_data = critique_selection_data[~np.in1d(np.ravel_multi_index(critique_selection_data.T, dims), np.ravel_multi_index(previous_critiques.T, dims))]
          #select critique from the adjusted data
          critique , head_or_tail = recommender.select_critique(critique_selection_data,args.critique_mode,recommended_items)
          if head_or_tail=="head":
            previous_critiques_head=np.vstack([previous_critiques_head,np.array([[ground_truth,critique[0],critique[1]]])])
          else:
            previous_critiques_tail=np.vstack([previous_critiques_tail,np.array([[critique[0],critique[1],ground_truth]])])
          if head_or_tail != None:
            if head_or_tail=="head":
              if args.critique_target == "object":
                _, true_relation_embedding, true_object_embedding = recommender.get_direct_embeddings(0,critique[0],critique[1])
                X_true=np.multiply((likes_embedding),(true_object_embedding.detach().numpy()))
              else:
                liked_items_list = recommender.obj2item(critique[1],kg_data)
                X_true = np.multiply((np.take(items_embeddings_tail,liked_items_list,axis=0)),(likes_embedding.detach().numpy()))
            else:
              if args.critique_target == "object":
                true_object_embedding, true_relation_embedding, _ = recommender.get_direct_embeddings(critique[0],critique[1],0)
                X_true=np.multiply((likes_embedding),(true_object_embedding.detach().numpy()))
              else:
                liked_items_list = recommender.obj2item(critique[0],kg_data)
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
            user_posterior=mu_prior
            user_posterior_proj = np.multiply(mu_prior,likes_embedding)
            _,Sigma_prior=updater.compute_laplace_approximation()
            recommended_items, rank= recommender.post_critiquing_recommendation(user_posterior_proj)
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

      wandb.log({
        "hits@1 session0":hitatone0,"hits@3 session0":hitatthree0,"hits@5 session0":hitatfive0,"hits@10 session0":hitatten0,"hits@20 session0":hitattwenty0,
"hits@1 session1":hitatone1,"hits@3 session1":hitatthree1,"hits@5 session1":hitatfive1,"hits@10 session1":hitatten1,"hits@20 session1":hitattwenty1,
"hits@1 session2":hitatone2,"hits@3 session2":hitatthree2,"hits@5 session2":hitatfive2,"hits@10 session2":hitatten2,"hits@20 session2":hitattwenty2,
"hits@1 session3":hitatone3,"hits@3 session3":hitatthree3,"hits@5 session3":hitatfive3,"hits@10 session3":hitatten3,"hits@20 session3":hitattwenty3,
"hits@1 session4":hitatone4,"hits@3 session4":hitatthree4,"hits@5 session4":hitatfive4,"hits@10 session4":hitatten4,"hits@20 session4":hitattwenty4,
"hits@1 session5":hitatone5,"hits@3 session5":hitatthree5,"hits@5 session5":hitatfive5,"hits@10 session5":hitatten5,"hits@20 session5":hitattwenty5
      })

          

            #get the rank of ground truth after update
            #measurer= Measure(user_embedding, likes_embedding, items_list, ground_truth, args.emb_dim)
            #ground_truth_rank= int(measurer.get_rank())
            #history[user_id][ground_truth].append(ground_truth_rank)
            #print(history)

    with open('saved_history.pkl', 'wb') as f:
      pickle.dump(history, f)



    #print("~~~~ Training ~~~~")
    #trainer = Trainer(dataset, args)
    #trainer.train()

    #print("~~~~ Select best epoch on validation set ~~~~")
    #epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
    #dataset = Dataset(args.dataset)
    
    #best_mrr = -1.0
    #best_epoch = "0"
    #for epoch in epochs2test:
    #    start = time.time()
    #    print(epoch)
     #   model_path = "models/" + args.dataset + "/" + epoch + ".chkpnt"
      #  tester = Tester(dataset, model_path, "valid")
       # mrr = tester.test()
        #if mrr > best_mrr:
         #   best_mrr = mrr
          #  best_epoch = epoch
        #print(time.time() - start)

   # print("Best epoch: " + best_epoch)

   # print("~~~~ Testing on the best epoch ~~~~")
   # best_model_path = "models/" + args.dataset + "/" + best_epoch + ".chkpnt"
   # tester = Tester(dataset, best_model_path, "test")
   # tester.test()
