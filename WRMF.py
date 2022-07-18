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
np.save('datasets/ML_FB/rec_test.npy', rec_test, allow_pickle=True)
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
def get_cold(matrix_coo, m, n):
    warm_rows = np.unique(matrix_coo.row)
    warm_cols = np.unique(matrix_coo.col)

    mask = np.ones(m, bool)
    mask[warm_rows] = 0
    cold_rows = np.nonzero(mask)

    mask = np.ones(n, bool)
    mask[warm_cols] = 0
    cold_cols = np.nonzero(mask)

    return cold_rows, cold_cols

# %%
from scipy.sparse import coo_matrix
row=rec_train_pd['users_id'].values
col=rec_train_pd['item_id'].values
data=np.ones(np.shape(row)[0])
matrix_train=coo_matrix((data, (row, col)))
# %%
m, n =matrix_train.shape
cold_rows, cold_cols = get_cold(matrix_train, m, n)
# %%
np.random.seed(2)
rank=200
iteration=10
alpha=1
lam=100
U = torch.tensor(np.random.normal(0, 0.01, size=(m, rank)).astype(np.float32)).float()
V = torch.tensor(np.random.normal(0, 0.01, size=(n, rank)).astype(np.float32)).float()
U[cold_rows] = 0
V[cold_cols] = 0
# %%
def per_item(vector_r, matrix_A, matrix_B, alpha):
    vector_r_index = torch.tensor(vector_r.nonzero()[0]).type(torch.long)
    vector_r_small = torch.tensor(vector_r[vector_r.nonzero()]).float()
    vector_c_small = alpha * vector_r_small
    matrix_B_small = matrix_B[vector_r_index]
    matrix_BT_small = torch.transpose(matrix_B_small, 0, 1)
    denominator = torch.inverse(matrix_A+torch.mm((torch.mul(matrix_BT_small, vector_c_small)), matrix_B_small))
    return torch.flatten(torch.mv(torch.mm(denominator, matrix_BT_small), torch.mul(vector_c_small, vector_r_small)+vector_r_small))

# %%
def solve(R, X, H, lam, rank, alpha):
    """
    Linear function solver, in the form R = XH^T with weighted loss
    """
    HT = torch.transpose(H, 0, 1)
    matrix_A = torch.mm(HT, H) + torch.eye(rank)*lam

    for i in tqdm(range(R.shape[1])):
        vector_r = R[:, i]
        vector_x = per_item(vector_r, matrix_A, H, alpha)
        X[i] = vector_x

# %%
matrix_input=matrix_train.toarray()
for i in range(iteration):
    solve(matrix_input.T, U, V, lam=lam, rank=rank, alpha=alpha)
    solve(matrix_input, V, U, lam=lam, rank=rank, alpha=alpha)

# %%
return0=U.numpy()
return1=V.numpy().T
return2=None
# %%
matrix_U , Yt, bias = return0, return1, return2
matrix_V =Yt.T

import cupy as cp
matrix_U = cp.array(matrix_U)
matrix_V = cp.array(matrix_V)
# %%
#TODO: this is testing on all train users we should just use indices that refer to test users
#for user_index in tqdm(range(matrix_U.shape[0])):
#    vector_u = matrix_U[user_index]
#    vector_train = matrix_input[user_index]
#    train_index = vector_train.nonzero()[0]
#    vector_predict = matrix_V.dot(vector_u)

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
