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
        self.hinge_margin = args.hinge_margin

        self.ent_h_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
        self.ent_t_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
        self.rel_embs     = nn.Embedding(self.num_rel, args.emb_dim).to(device)
        self.rel_inv_embs = nn.Embedding(self.num_rel, args.emb_dim).to(device)

        weights = [self.ent_h_embs.weight.data, self.ent_t_embs.weight.data, 
                   self.rel_embs.weight.data, self.rel_inv_embs.weight.data] 

        # set reduce type
        if args.reduce_type == 'mean':
            self.reduce = torch.mean
        elif args.reduce_type == 'sum':
            self.reduce = torch.sum

        # init weights
        if args.init_type == 'uniform':
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
        # TODO: remove if statement and make
        # self.loss_fc(score, label) in __init__
        if self.loss_type == 'softplus':
            out = F.softplus(-labels * score)

        elif self.loss_type == 'gauss':
            out = torch.square(score - labels)
        elif self.loss_type == 'hinge':
            l = (-labels * score + self.hinge_margin)
            zeros = torch.zeros(l.shape, device= self.device)
            mask = l > zeros
            out = l[mask]
        elif self.loss_type == 'PSL':
            l = (-labels * score + self.hinge_margin)
            zeros = torch.zeros(l.shape, device= self.device)
            mask = l > zeros
            l = l[mask]
            out = 0.5 * torch.square(l)

        return self.reduce(out)

    def reg_loss(self):
        ent_dim = self.ent_h_embs.weight.shape[0]
        rel_dim = self.rel_embs.weight.shape[0]
 
        if self.reg_type == 'gauss':
            node = torch.zeros(ent_dim).to(self.device)
            edge = torch.zeros(rel_dim).to(self.device)
        elif self.reg_type == 'tilt' in self.reg_type:
            node = torch.ones(ent_dim).to(self.device)
            edge = torch.sqrt(torch.tensor(self.emb_dim)) * torch.ones(rel_dim)
            edge = edge.to(self.device)
            
        reg_h = self.reduce((torch.linalg.norm(self.ent_h_embs.weight, axis=1) - node) ** 2) 
        reg_t = self.reduce((torch.linalg.norm(self.ent_t_embs.weight, axis=1) - node) ** 2)
        reg_rel_f = self.reduce((torch.linalg.norm(self.rel_embs.weight, axis=1) - edge) ** 2)
        reg_rel_inv = self.reduce((torch.linalg.norm(self.rel_inv_embs.weight, axis=1) - edge) ** 2)

        norm_val = torch.tensor([reg_h, reg_t, reg_rel_f, reg_rel_inv])
        return self.reduce(norm_val)
