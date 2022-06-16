import torch
import os, sys
import numpy as np
from utils.plots import rank_save

def test(model, dataloader, args, device):
    # get items and users 
    items = dataloader.items.tolist()
    users = dataloader.test_users.tolist()
    user_likes = dataloader.user_likes_map

    # to store embeddings and projected embeddings (head and tail)
    items_h = np.zeros([len(items), args.emb_dim])
    items_t = np.zeros([len(items), args.emb_dim])
    users_hp = np.zeros([len(users), args.emb_dim])
    users_tp = np.zeros([len(users), args.emb_dim])

    # to map id to array location
    item_id2index = dict(zip(items, list(range(0, len(items)))))

    # save embeddings in array
    for item in items:
        index = item_id2index[item]
        item_temp = torch.tensor([item]).long()

        items_h[index] = model.ent_h_embs(item_temp).detach().numpy()
        items_t[index] = model.ent_t_embs(item_temp).detach().numpy()
    
    r = torch.tensor([dataloader.likes_link]).long()
    likes_f = model.rel_embs(r).detach().numpy()
    likes_inv = model.rel_inv_embs(r).detach().numpy()

    # to map id to array location
    user_id2index = dict(zip(users, list(range(0, len(items)))))

    # TODO: shuffle?
    for user in users:
        index = user - users[0]
        user_temp = torch.tensor([user]).long()
        
        new_h = model.ent_h_embs(user_temp).detach().numpy()
        new_t = model.ent_t_embs(user_temp).detach().numpy()
        users_hp[index] = new_h * likes_f
        users_tp[index] = new_t * likes_inv

    # main test loop
    rank_track = []
    for i, user in enumerate(user_likes.keys()):
        if i > 100: break
        index = user - users[0]

        for_prod = np.sum(users_hp[index] * items_t, axis=1)
        inv_prod = np.sum(users_tp[index] * items_h, axis=1)

        scores = np.clip((for_prod + inv_prod) / 2, -40, 40)
        ranked = scores.argsort()[::-1]

        for item in user_likes[user]:
            rank = int(np.where(ranked == item_id2index[item])[0])
            rank_track.append(rank)

    rank_save(rank_track, args.test_name, shuffle=True)
