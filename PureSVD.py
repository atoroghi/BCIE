
import os, pickle, sys, time
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import tqdm
import statistics
import scipy.sparse as sp
from scipy.sparse import vstack
from sklearn.utils.extmath import randomized_svd
import cupy as cp

class PureSVD:
    def __init__(self, dataloader, args):
        self.rec_train = dataloader.rec_train
        self.rec_test = dataloader.rec_test
        self.user_likes_mine = dataloader.user_likes_map
        self.user_likes_whole = dataloader.user_likes_whole
        self.users = (pd.unique(self.rec_train[:,0])).tolist()
        self.items = (pd.unique(self.rec_train[:,2])).tolist()
        self.train_items = np.unique(self.rec_train[:,2])
        self.test_items = np.unique(self.rec_test[:,2])
        self.rank = args.emb_dim
        self.epochs = args.epochs
        self.item2index = dict(zip(self.items, list(range(0, len(self.items)))))
        self.user2index = dict(zip(self.users, list(range(0, len(self.users)))))

    def train_model(self):
        user_ids_train=[]
        item_ids_train=[]
        rec_train_pd = pd.DataFrame(self.rec_train,columns=["users","likes","item"])
        for i,row in rec_train_pd.iterrows():
            user_ids_train.append(self.user2index[row['users']])
            item_ids_train.append(self.item2index[row['item']])

        rec_train_pd['users_id']=user_ids_train
        rec_train_pd['item_id']=item_ids_train
        indices = torch.LongTensor(
            rec_train_pd[['users_id', 'item_id']].values
        )
        values = torch.ones(indices.shape[0])
        sparse = torch.sparse.FloatTensor(indices.t(), values)
        pred_df = rec_train_pd
        pred_df = pred_df[['users_id']].drop_duplicates()
        _user_tensor = sparse.to_dense().index_select(
                    dim=0, index=torch.LongTensor(pred_df['users_id'])
                )
        row=rec_train_pd['users_id'].values
        col=rec_train_pd['item_id'].values
        data=np.ones(np.shape(row)[0])
        matrix_input=coo_matrix((data, (row, col)))
        seed=2
        P, sigma, Qt = randomized_svd(matrix_input,
                                            n_components=self.rank,
                                            n_iter=self.epochs,
                                            power_iteration_normalizer='QR',
                                            random_state=seed)

        RQ = matrix_input.dot(sp.csc_matrix(Qt).T)
        matrix_U , Yt, bias = np.array(RQ.todense()), Qt, None
        matrix_V =Yt.T
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

        return matrix_U, matrix_V

    def vector_delete(self,vector,indices):
        filtered = 1*vector
        filtered[indices]= -100
        return filtered

    def test_model(self, matrix_U, matrix_V):
        ranks=[]
        test_users_list=list(self.user_likes_mine.keys())[:1000]
        for test_user in tqdm(test_users_list):
            test_user_id = self.user2index[test_user]
            vector_u = matrix_U[test_user_id]
            vector_predict = matrix_V.dot(vector_u)
            for liked_item in self.user_likes_mine[test_user]:
                item_index = self.item2index[liked_item]
                other_liked_items=(self.user_likes_whole[test_user])
                other_liked_items.remove(liked_item)
                other_liked_items_indices=[self.item2index[item] for item in other_liked_items]
                preds_filtered = self.vector_delete(vector_predict,other_liked_items_indices)
                ranked = preds_filtered.argsort()[::-1]
                rank = int(np.where(ranked == item_index)[0])
                ranks.append(rank)
        return np.array(ranks)