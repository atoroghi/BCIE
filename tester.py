import torch
import os, sys
import numpy as np
from utils.plots import rank_save

def test(dataloader, args, device):
    # load model
    path = os.path.join('results', args.test_name)
    load_path = os.path.join(path, 'models', 'epoch {}.chkpnt'.format(args.epochs))
    model = torch.load(load_path, map_location = 'cpu')

    # get items and users 
    items = dataloader.items.tolist()
    users = dataloader.users.tolist()
    user_likes = dataloader.user_likes_map

    # to store embeddings and projected embeddings (head and tail)
    items_h = np.zeros([len(items), args.emb_dim])
    items_t = np.zeros([len(items), args.emb_dim])
    users_hp = np.zeros([len(users), args.emb_dim])
    users_tp = np.zeros([len(users), args.emb_dim])

    # to map id to array location
    id2index = dict(zip(items, list(range(0, len(items)))))

    # save embeddings in array
    for item in items:
        index = id2index[item]
        item_temp = torch.tensor([item]).long()

        items_h[index] = model.ent_h_embs(item_temp).detach().numpy()
        items_t[index] = model.ent_t_embs(item_temp).detach().numpy()
    
    r = torch.tensor([dataloader.likes_link]).long()
    likes_f = model.rel_embs(r).detach().numpy()
    likes_inv = model.rel_inv_embs(r).detach().numpy()
    for user in users:
        index = user - users[0]
        user_temp = torch.tensor([user]).long()
        
        new_h = model.ent_h_embs(user_temp).detach().numpy()
        new_t = model.ent_t_embs(user_temp).detach().numpy()
        
        users_hp[index] = new_h * likes_f
        users_tp[index] = new_t * likes_inv

    # main test loop
    #################################################################
    i = 0
    rank_track = []
    for user in user_likes.keys():
        if i > 1000: break
        index = user - users[0]

        for_prod = np.sum(users_hp[index] * items_t, axis=1)
        inv_prod = np.sum(users_tp[index] * items_h, axis=1)

        scores = np.clip((for_prod + inv_prod) / 2, -40, 40)
        ranked = scores.argsort()[::-1]

        for ground_truth in user_likes[user]:
            rank = int(np.where(ranked == id2index[ground_truth])[0])
            rank_track.append(rank)
        i += 1

    rank_save(rank_track, args.test_name)
