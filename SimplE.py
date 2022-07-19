import torch, sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SimplE(nn.Module):
    def __init__(self, dataloader, args, device):
        super(SimplE, self).__init__()
        self.num_ent = dataloader.num_item
        self.num_rel = dataloader.num_rel
        self.emb_dim = args.emb_dim
        self.reg_lambda = args.reg_lambda
        self.reg_type = args.reg_type
        self.loss_type = args.loss_type
        self.device = device

        self.ent_h_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
        self.ent_t_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
        self.rel_embs     = nn.Embedding(self.num_rel, args.emb_dim).to(device)
        self.rel_inv_embs = nn.Embedding(self.num_rel, args.emb_dim).to(device)

        weights = [self.ent_h_embs.weight.data, self.ent_t_embs.weight.data, 
                   self.rel_embs.weight.data, self.rel_inv_embs.weight.data] 

        # init weights
        if args.init_type == 'uniform':
            #args.init_scale = sqrt_size = 6.0 / np.sqrt(self.emb_dim)
            for w in weights:
                nn.init.uniform_(w, -args.init_scale, args.init_scale)

        elif args.init_type == 'normal':
            for w in weights:
                nn.init.normal_(w, std=args.init_scale)

    def forward(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        r_embs = self.rel_embs(rels)
        r_inv_embs = self.rel_inv_embs(rels)

        for_prod = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        inv_prod = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((for_prod + inv_prod) / 2, -20, 20) 

    def loss(self, score, labels):
        if self.loss_type == 'softplus':
            out = F.softplus(-labels * score)
            loss = torch.sum(out)

        elif self.loss_type == 'gauss':
            out = torch.sigmoid(-labels * score)
            loss = torch.mean(out)

        return loss

    def reg_loss(self):
        if self.reg_type == 'gauss':
            norm_val = (
                  (torch.norm(self.ent_h_embs.weight, p=2) ** 2) \
                + (torch.norm(self.ent_t_embs.weight, p=2) ** 2) \
                + (torch.norm(self.rel_embs.weight, p=2) ** 2) \
                + (torch.norm(self.rel_inv_embs.weight, p=2) ** 2))

            return norm_val

        elif 'tilt' in self.reg_type:
            ent_dim = self.ent_h_embs.weight.shape[0]
            rel_dim = self.rel_embs.weight.shape[0]
            one = torch.ones(ent_dim).to(self.device)
            root = torch.sqrt(torch.tensor(self.emb_dim)) * torch.ones(rel_dim)
            root = root.to(self.device)

            if self.reg_type == 'tilt_mean':
                op = torch.mean
            elif self.reg_type == 'tilt_sum':
                op = torch.sum
            
            reg_h = op((torch.linalg.norm(self.ent_h_embs.weight, axis=1) - one) ** 2) 
            reg_t = op((torch.linalg.norm(self.ent_t_embs.weight, axis=1) - one) ** 2)
            reg_rel_f = op((torch.linalg.norm(self.rel_embs.weight, axis=1) - root) ** 2)
            reg_rel_inv = op((torch.linalg.norm(self.rel_inv_embs.weight, axis=1) - root) ** 2)

            norm_val = reg_h + reg_t + reg_rel_f + reg_rel_inv
            return norm_val
