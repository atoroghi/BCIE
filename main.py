from trainer import Trainer
from tester import Tester
from dataset import Dataset
from measure import Measure
from recommender import Recommender
from updater import Updater
from plotter import Plotter
import matplotlib.pyplot as plt
import argparse
import time
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=1000, type=int, help="number of epochs")
    parser.add_argument('-ni', default=0, type=float, help="noise intensity")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="WN18", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=200, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=1415, type=int, help="batch size")
    parser.add_argument('-save_each', default=50, type=int, help="validate every k epochs")
    parser.add_argument('-max_iters_laplace', default=1000, type=int, help="Maximum number of iterations for Laplace Approximation")
    parser.add_argument('-alpha', default=0.01, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-etta', default=1.0, type=float, help="Learning rate for Laplace Approximation")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    dataset = Dataset(args.dataset,args)
    model_path="models/" + "MLraw" + "/" + "2000" + ".chkpnt"
    #ground_truth=4
    users_list= dataset.users
    items_list= dataset.items
    users_likes= dataset.users_likes

    history={}
    
    session_length= 3


    #critiquing loop:
    for user_id in users_list:
      print("user_id:"+str(user_id))
      history[user_id]={}
      print("users_likes"+str(users_likes[user_id]))
      user_posterior=torch.ones(args.emb_dim)
      for ground_truth in users_likes[user_id]:
        print("ground_truth"+str(ground_truth))
        history[user_id][ground_truth]=[]
        previous_critiques={"head":[],"tail":[]}
        # pre-critiquing recommendation. Output user embedding, the list of initially recommended items, and the rank of ground truth
        recommender= Recommender(dataset,model_path,user_id,ground_truth,"pre",user_posterior)
        user_embedding, likes_embedding, recommended_items, initial_rank= recommender.pre_critiquing_recommendation()
        print("recommended_items"+str(recommended_items))
        history[user_id][ground_truth].append(initial_rank)

        for session_no in range (0,session_length):
          print("session_no"+str(session_no))
          truth_critiquing_candidate, recommended_critiquing_candidate= recommender.get_critique_candidates()
          print("truth_critiquing_candidate:"+str(truth_critiquing_candidate))
          print("rec_crtqng_cnd"+str(recommended_critiquing_candidate))
          #frequent= recommender.get_frequent()
          #print("frequent"+str(frequent))
          head_or_tail,critique_rel,critique_destination,false_facts= recommender.select_critique(previous_critiques)
          if head_or_tail=="head":
            previous_critiques[head_or_tail].append([critique_rel,critique_destination])
          else:
            previous_critiques[head_or_tail].append([critique_destination,critique_rel])
          
          print("previous_critiques:"+str(previous_critiques))
          if head_or_tail != None:
            X_false= torch.empty(0)
            X_true= torch.empty(0)
            #_, true_relation_embedding, true_item_embedding = recommender.get_direct_embeddings(0,critique_rel,1.0)
            if head_or_tail=="head":
              _, true_relation_embedding, true_item_embedding = recommender.get_direct_embeddings(0,critique_rel,critique_destination)
              #X_true= torch.cat([X_true,true_relation_embedding*true_item_embedding])
              X_true= torch.cat([X_true,likes_embedding*true_item_embedding])
              for fact in false_facts:
                _, false_relation_embedding, false_item_embedding = recommender.get_direct_embeddings(0,fact[0],fact[1])
                #X_false= torch.cat([X_false,false_relation_embedding*false_item_embedding])
                X_false= torch.cat([X_false,likes_embedding*false_item_embedding])
            else:
              true_item_embedding, true_relation_embedding, _ = recommender.get_direct_embeddings(critique_destination,critique_rel,0)
              #X_true= torch.cat([X_true,true_item_embedding*true_relation_embedding])
              X_true= torch.cat([X_true,true_item_embedding*likes_embedding])
              for fact in false_facts:
                false_item_embedding, false_relation_embedding, _= recommender.get_direct_embeddings(fact[1],fact[0],0)
                #X_false= torch.cat([X_false,false_item_embedding*false_relation_embedding])
                X_false= torch.cat([X_false,false_item_embedding*likes_embedding])
            

            y= -1*torch.ones((X_true.size()[0])+(X_false.size()[0]))
            X_all= torch.cat([X_true,X_false])
            y[0:(X_true.size()[0])]= 1
       
          

          #performing Laplace Approximation

            max_iters= args.max_iters_laplace
            alpha= args.alpha
            etta= args.etta
            user_embedding= user_embedding.view(args.emb_dim)
            #print("user_embedding:"+str(user_embedding))
            W=user_embedding
            if session_no==0:
              Sigma_prior= 100*torch.eye(args.emb_dim)


            updater=Updater(X_all,y,user_embedding,Sigma_prior,W,args)
            b=updater.compute_laplace_approximation()
            mu_prior=torch.tensor([float(b[0][i]) for i in range(0,args.emb_dim)], requires_grad=True, dtype=torch.float)
            user_embedding= mu_prior.detach()
            Sigma_prior_new= b[1]
            Sigma_prior= Sigma_prior.detach()
            #print("user_embedding"+str(user_embedding))

            recommender= Recommender(dataset,model_path,user_id,ground_truth,"post",user_embedding)
            _, likes_embedding, recommended_items, rank= recommender.post_critiquing_recommendation()
            history[user_id][ground_truth].append(rank)


          

            #get the rank of ground truth after update
            #measurer= Measure(user_embedding, likes_embedding, items_list, ground_truth, args.emb_dim)
            #ground_truth_rank= int(measurer.get_rank())
            #history[user_id][ground_truth].append(ground_truth_rank)
            print(history)

    #print("user_updated")
    
    plotter= Plotter(history,session_length)
    plotter.plot_ranks()







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
