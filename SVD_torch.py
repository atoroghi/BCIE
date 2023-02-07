import sys, torch
from tester import GetGT, RankTrack, get_rank, get_Rprec
from proc import adj_matrix
import numpy as np
from utils import rank_plot, save_metrics


def svd(dataloader, args, mode, device):
    a, u_min = adj_matrix(args.fold, args.dataset) # adjacency matrix of user item
    a = a.to('cuda')
    m,n = a.shape
    
    R = a.to_dense()

    P, sigma, Qt = torch.svd_lowrank(a, args.rank, niter=args.n_iter)
    RQ = torch.mm(R, Qt)

    # train
    print('args: {} {} {} {}'.format(args.rank, args.n_iter, args.alpha, args.lam))
    out = RQ @ Qt.T

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
        #rprec = get_Rprec(ranked, test_gt, train_gt, id2index[i])

        rank_track.update(ranks, 1, 0)

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