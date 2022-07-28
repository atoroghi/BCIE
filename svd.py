import sys, torch
from tester import GetGT, RankTrack, get_rank 
from proc import adj_matrix
import numpy as np
from utils import rank_plot, save_metrics

def svd(dataloader, args, mode, device):
    a = adj_matrix(args.fold) # adjacency matrix of user item
    a = a.to('cuda')

    # train
    (u, s, v) = torch.svd_lowrank(a, q=100, niter=2)
    print('train done')
    e = s * torch.eye(s.shape[0]).to(device)
    out = u @ (e @ v.T)

    # all users and items in maps
    data = dataloader.rec_test if mode == 'test' else dataloader.rec_val
    all_users = torch.tensor(data[:,0]).to('cuda')
    users = torch.unique(all_users, sorted=False).to('cpu').tolist()
    user2index = dict(zip(users, list(range(len(users)))))
    
    data = dataloader.rec_train
    all_items = torch.tensor(data[:,2]).to('cuda')
    items = torch.unique(all_items, sorted=False).to('cpu').tolist()
    id2index = dict(zip(items, list(range(len(items)))))

    # load gt info and track rank scores classes
    get_gt = GetGT(dataloader.fold, mode)
    rank_track = RankTrack()

    # main test loop
    print(out.shape)
    for j, user in enumerate(users):
        if j > 1000: break
        scores = out[user2index[user]]
        ranked = torch.argsort(scores, descending=True)

        test_gt, all_gt = get_gt.get(user)
        if test_gt == None: continue
        
        ranks = get_rank(ranked, test_gt, all_gt, id2index)
        rank_track.update(ranks, 0)

    # different save options if train or testing        
    epoch = 0
    if mode == 'val': 
        rank_plot(rank_track, args.test_name, epoch)
        rank_at_k = save_metrics(rank_track, args.test_name, epoch, mode)
        return rank_at_k

    else:
        save_metrics(rank_track, args.test_name, epoch, mode)
