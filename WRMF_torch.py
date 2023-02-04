import sys, torch
from tester import GetGT, RankTrack, get_rank, get_Rprec
from proc import adj_matrix
import numpy as np
from utils import rank_plot, save_metrics

def get_cold(s, m, n):
    
    warm_rows=torch.unique(s._indices()[0])
    warm_cols=torch.unique(s._indices()[1])
    mask = torch.ones(m,dtype=bool)
    mask[warm_rows] = 0
    cold_rows = torch.nonzero(mask)
    mask = torch.ones(n,dtype=bool)
    mask[warm_cols] = 0
    cold_cols = torch.nonzero(mask)
    return cold_rows, cold_cols

def per_item(vector_r, matrix_A, matrix_B, alpha):
    vector_r_index = vector_r.nonzero().type(torch.long).flatten()
    vector_r_small = vector_r[vector_r.nonzero()].float().flatten()
    vector_c_small = alpha * vector_r_small
    matrix_B_small = matrix_B[vector_r_index]
    matrix_BT_small = torch.transpose(matrix_B_small, 0, 1)
    denominator = torch.inverse(matrix_A+torch.mm((torch.mul(matrix_BT_small, vector_c_small)), matrix_B_small))
    return torch.flatten(torch.mv(torch.mm(denominator, matrix_BT_small), torch.mul(vector_c_small, vector_r_small)+vector_r_small))

def solve(R, X, H, lam, rank, alpha):
    """
    Linear function solver, in the form R = XH^T with weighted loss
    """
    HT = torch.transpose(H, 0, 1)
    lam_tensor = (lam*torch.eye(rank).to('cuda'))
    matrix_A = torch.mm(HT, H) + lam_tensor
    

    for i in (range(R.shape[1])):
        vector_r = R[:, i]
        vector_x = per_item(vector_r, matrix_A, H, alpha)
        X[i] = vector_x

def wrmf(dataloader, args, mode, device):
    a, u_min = adj_matrix(args.fold, args.dataset) # adjacency matrix of user item
    a = a.to('cuda')
    m,n = a.shape
    cold_rows, cold_cols = get_cold(a, m, n)
    U = torch.tensor(np.random.normal(0, 0.01, size=(m, args.rank)).astype(np.float32)).float().to('cuda')
    V = torch.tensor(np.random.normal(0, 0.01, size=(n, args.rank)).astype(np.float32)).float().to('cuda')
    U[cold_rows] = 0
    V[cold_cols] = 0
    R = a.to_dense()
    for i in range(args.n_iter):
        solve(R.T, U, V, lam= args.lam, rank= args.rank, alpha= args.alpha)
        solve(R, V, U, lam= args.lam, rank= args.rank, alpha= args.alpha)


    # train
    print('args: {} {} {} {}'.format(args.rank, args.n_iter, args.alpha, args.lam))
    out = U @ V.T

    #(u, s, v) = torch.svd_lowrank(a, q=args.rank, niter=args.n_iter)
    #e = s * torch.eye(s.shape[0]).to(device)
    #out = u @ (e @ v.T) # estimate of user likes matrix
    
    # all users and items in maps
    data = dataloader.rec_test if mode == 'test' else dataloader.rec_val
    all_users = torch.tensor(data[:,0]).to('cuda')
    users = torch.unique(all_users, sorted=False).to('cpu').tolist()
    user2index = {}
    for u in users:
        user2index.update({u : u - u_min})
    
    data = dataloader.rec_train
    all_items = torch.tensor(data[:,2]).to('cuda')
    items = torch.unique(all_items, sorted=False).to('cpu').tolist()
    id2index = {}
    for it in items:
        id2index.update({it : it})

    # load gt info and track rank scores classes
    get_gt = GetGT(dataloader.fold, mode, args.dataset)
    rank_track = RankTrack()

    # main test loop
    for j, user in enumerate(users):
        scores = out[user2index[user]]
        ranked = torch.argsort(scores, descending=True)

        test_gt, all_gt, train_gt = get_gt.get(user)
        if test_gt == None: continue
        
        ranks = get_rank(ranked, test_gt, all_gt, id2index)
        rprec = get_Rprec(ranked, test_gt, train_gt, id2index[i])

        rank_track.update(ranks, rprec, 0)

    # different save options if train or testing        
    epoch = 0
    rank_plot(rank_track, args.test_name, epoch)
    save_metrics(rank_track, args.test_name, epoch, mode)
    #if mode == 'val': 
    #    rank_plot(rank_track, args.test_name, epoch)
    #    rank_at_k = save_metrics(rank_track, args.test_name, epoch, mode)
    #    return rank_at_k
##
#    #else:
    #    save_metrics(rank_track, args.test_name, epoch, mode)