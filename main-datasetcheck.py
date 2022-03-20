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
import numpy as np
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
    dataset = Dataset(args.dataset,args.ni)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path="Noisecheckmodels/" + "noise0.2" + "/" + "5000" + ".chkpnt"
    #model_path="models/" + "ML" + "/" + "10000" + ".chkpnt"
    model = torch.load(model_path, map_location = device)
    model.eval()
    #ground_truth=4
    users_list= dataset.users
    items_list= dataset.items
    users_likes= dataset.users_likes

    user_recommendations={}
    

    #for user_id in users_list:

     #      h=torch.tensor([user_id]).long().to(device)
      #     r=torch.tensor([0]).long().to(device)
       #    scores_dict={}
        #   for item in items_list:
         #      t=torch.tensor([item]).long().to(device)
          #     score,user_embedding,likes_embedding,item_embedding,_,_,_=model(h,r,t)
           #    scores_dict[item]=score
           #recommended_items=sorted(scores_dict, key=scores_dict.get, reverse=True)
           #user_recommendations[user_id]=recommended_items


    #print(user_recommendations)
    for user_id in users_list:
        new_mu_prior=np.zeros((args.emb_dim))
        new_sigma_prior=10*np.eye(args.emb_dim)
        new_user_embedding=np.random.multivariate_normal(new_mu_prior, new_sigma_prior)
        new_user_embedding_tens=torch.tensor(new_user_embedding)
        new_user_embedding_tens=new_user_embedding_tens.view(args.emb_dim)
        scores_dict={}
        for item in items_list:
            t_emb=torch.tensor([item]).long().to(device)
            r_emb=torch.tensor([0]).long().to(device)
            h_emb=torch.tensor([user_id]).long().to(device)
            _,_,likes_embedding,item_embedding,ht_embs , r_inv_embs , th_embs=model(h_emb,r_emb,t_emb)
            scores1=torch.sum(new_user_embedding_tens * likes_embedding * item_embedding, dim=1)
            scores2=torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
            score=torch.clamp((scores1+scores2)/2, -20, 20)
            scores_dict[item]=score
        recommended_items=sorted(scores_dict, key=scores_dict.get, reverse=True)
        user_recommendations[user_id]=recommended_items
    print(user_recommendations)


    user_critiques={0:484,5:42,10:293,15:137}
    scores_dict_post={}
    user_recommendations_post={}
    for user_id in user_critiques.keys():
        #user_posterior=torch.ones(args.emb_dim)
        critique=user_critiques[user_id]
        h=torch.tensor([user_id]).long().to(device)
        t=torch.tensor([critique]).long().to(device)
        r=torch.tensor([0]).long().to(device)
        _,user_embedding,likes_embedding,item_embedding,_,likes_embedding_inv,item_embedding_inv=model(h,r,t)
        new_mu_prior=np.zeros((args.emb_dim))
        new_sigma_prior=10*np.eye(args.emb_dim)
        new_user_embedding=np.random.multivariate_normal(new_mu_prior, new_sigma_prior)
        new_user_embedding_tens=torch.tensor(new_user_embedding)
        X_true= torch.empty(0).to(device)
        X_true= torch.cat([X_true,likes_embedding*item_embedding])
        y= torch.ones((X_true.size()[0]))
        max_iters = args.max_iters_laplace
        alpha= args.alpha
        etta= args.etta
        #user_embedding= user_embedding.view(args.emb_dim)
        #print("user_embedding_shape"+str(user_embedding.shape))
        new_user_embedding_tens=new_user_embedding_tens.view(args.emb_dim)
        #user_embedding=user_embedding.detach().numpy()
        Sigma_prior= 100*torch.eye(args.emb_dim)
        #updater=Updater(X_true, y,user_embedding, Sigma_prior, user_embedding ,args)
        updater=Updater(X_true, y,new_user_embedding_tens, Sigma_prior, new_user_embedding_tens ,args)
        #b=updater.compute_laplace_approximation()
        with torch.no_grad():
            #mu_prior, _ =updater.SDR_cvxopt(Sigma_prior,X_true,y,user_embedding)
            mu_prior, _ =updater.SDR_cvxopt(Sigma_prior,X_true,y,new_user_embedding_tens)
            user_posterior=torch.tensor(mu_prior)
        

        #user_posterior=torch.zeros(args.emb_dim)

        #mu_prior=torch.tensor([float(b[0][i]) for i in range(0,args.emb_dim)], requires_grad=True, dtype=torch.float)
        #user_embedding= mu_prior.detach()
        #Sigma_prior_new= b[1]
        #Sigma_prior= Sigma_prior_new.detach()
        for item in items_list:
            t_emb=torch.tensor([item]).long().to(device)
            r_emb=torch.tensor([0]).long().to(device)
            h_emb=torch.tensor([user_id]).long().to(device)
            _,_,likes_embedding,item_embedding,ht_embs , r_inv_embs , th_embs=model(h_emb,r_emb,t_emb)
            scores1=torch.sum(user_posterior * likes_embedding * item_embedding, dim=1)
            scores2=torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
            score=torch.clamp((scores1 + scores2) / 2, -20, 20)
            scores_dict_post[item]=score
        recommended_items_post=sorted(scores_dict_post, key=scores_dict_post.get, reverse=True)
        user_recommendations_post[user_id]=recommended_items_post

    print(user_recommendations_post)

