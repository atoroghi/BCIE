# %%
import os, pickle, sys, time
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
import statistics
# %%
path = os.path.join('datasets', 'ML_FB')
rec_train=np.load(os.path.join(path, 'rec_train.npy'), allow_pickle=True)
rec_test=np.load(os.path.join(path, 'rec_test.npy'), allow_pickle=True)
users = np.unique(rec_train[:,0])
train_items=np.unique(rec_train[:,2])
test_items=np.unique(rec_test[:,2])
invalid_items=[item for item in test_items if item not in train_items]
for invalid_item in invalid_items:
  rec_test=np.delete(rec_test,np.where(rec_test[:,2]==invalid_item),axis=0)
train_users=np.unique(rec_train[:,0])
test_users=np.unique(rec_test[:,0])
invalid_users=[user for user in test_users if user not in train_users]
for invalid_user in invalid_users:
  rec_test=np.delete(rec_test,np.where(rec_test[:,0]==invalid_user),axis=0)
np.shape(rec_test)
# %%
user_likes_mine={}
for user in np.unique(rec_test[:,0]):
    user_likes_mine[user]=list(rec_test[np.where(rec_test[:,0]==user),2][0])
# %%
import pickle
with open('user_likes_mine.pkl', 'wb') as f:
    pickle.dump(user_likes_mine, f)

# %%
import pickle
with open('user_likes_mine.pkl', 'rb') as f:
    user_likes_mine=pickle.load(f)
# %%
users=(pd.unique(rec_train[:,0])).tolist()
items=(pd.unique(rec_train[:,2])).tolist()
item2index = dict(zip(items, list(range(0, len(items)))))
user2index = dict(zip(users, list(range(0, len(users)))))
item2index_inv = dict(zip(list(range(0,len(items))),items))
item2index_inv = dict(zip(list(range(0,len(users))),users))

# %%
rec_train_pd=pd.DataFrame(rec_train,columns=["users","likes","item"])
rec_test_pd=pd.DataFrame(rec_test,columns=["users","likes","item"])
# %%
user_ids_train=[]
item_ids_train=[]
for i,row in rec_train_pd.iterrows():
    user_ids_train.append(user2index[row['users']])
    item_ids_train.append(item2index[row['item']])

rec_train_pd['users_id']=user_ids_train
rec_train_pd['item_id']=item_ids_train

# %%
user_ids_test=[]
item_ids_test=[]
for i,row in rec_test_pd.iterrows():
    user_ids_test.append(user2index[row['users']])
    item_ids_test.append(item2index[row['item']])

rec_test_pd['users_id']=user_ids_test
rec_test_pd['item_id']=item_ids_test
# %%
indices = torch.LongTensor(
            rec_train_pd[['users_id', 'item_id']].values
        )
values = torch.ones(indices.shape[0])
sparse = torch.sparse.FloatTensor(indices.t(), values)
# %%
pred_df = rec_train_pd
pred_df = pred_df[['users_id']].drop_duplicates()
_user_tensor = sparse.to_dense().index_select(
            dim=0, index=torch.LongTensor(pred_df['users_id'])
        )
# %%     
from scipy.sparse import coo_matrix
row=rec_train_pd['users_id'].values
col=rec_train_pd['item_id'].values
data=np.ones(np.shape(row)[0])
matrix_train=coo_matrix((data, (row, col)))
# %% 
import scipy.sparse as sparse
from scipy.sparse import vstack
from sklearn.utils.extmath import randomized_svd
matrix_input = matrix_train
iteration=10
rank=50
seed=2
P, sigma, Qt = randomized_svd(matrix_input,
                                      n_components=rank,
                                      n_iter=iteration,
                                      power_iteration_normalizer='QR',
                                      random_state=seed)

RQ = matrix_input.dot(sparse.csc_matrix(Qt).T)

matrix_U , Yt, bias = np.array(RQ.todense()), Qt, None
matrix_V =Yt.T

import cupy as cp
matrix_U = cp.array(matrix_U)
matrix_V = cp.array(matrix_V)

# %%
with open('user_likes_mine.pkl', 'rb') as f:
    user_likes_mine=pickle.load(f)
with open('user_likes_mine_whole.pkl', 'rb') as f:
    user_likes_mine_whole=pickle.load(f)

def th_delete_mine(tensor, indices):
    mask = torch.ones(tensor.numel())
    mask[indices] = -100
    return tensor*mask

def vector_delete(vector,indices):
    filtered = 1*vector
    filtered[indices]=-100
    return filtered
# %%
k=3

hit1s=[]
hit3s=[]
hit10s=[]
test_users_list=list(user_likes_mine.keys())
for test_user in tqdm(test_users_list):
    test_user_id = user2index[test_user]
    vector_u = matrix_U[test_user_id]
    vector_predict = matrix_V.dot(vector_u)
    for liked_item in user_likes_mine[test_user]:
        item_index = item2index[liked_item]
        other_liked_items=(user_likes_mine_whole[test_user])
        other_liked_items.remove(liked_item)
        other_liked_items_indices=[item2index[item] for item in other_liked_items]
        #preds_filtered = th_delete_mine(vector_predict,other_liked_items_indices)
        preds_filtered = vector_delete(vector_predict,other_liked_items_indices)
        #rec_indices_k = preds_filtered.topk(k).indices.tolist()
        ranked = preds_filtered.argsort()[::-1]
        rank = int(np.where(ranked == item_index)[0])
        if rank<2:
            hit1s.append(1)
        else:
            hit1s.append(0)
        if rank<4:
            hit3s.append(1)
        else:
            hit3s.append(0)
        if rank<11:
            hit10s.append(1)
        else:
            hit10s.append(0)

print("1:")
hitsmean1=statistics.mean(list(hit1s))
print("hits @ 1",hitsmean1)
print("CI of hits@1",1.96*(np.std(list(hit1s)))/np.sqrt(len(hit1s)))
print("3:")
hitsmean3=statistics.mean(list(hit3s))
print("hits @ 3",hitsmean3)
print("CI of hits@3",1.96*(np.std(list(hit3s)))/np.sqrt(len(hit3s)))
print("10:")
hitsmean10=statistics.mean(list(hit10s))
print("hits @ 10",hitsmean10)
print("CI of hits@10",1.96*(np.std(list(hit10s)))/np.sqrt(len(hit10s)))
# %%
