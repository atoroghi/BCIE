import torch, sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ComplEx(nn.Module):
    def __init__(self, dataloader, args, device):
        super(ComplEx, self).__init__()
        self.num_ent = dataloader.num_item
        self.num_rel = dataloader.num_rel
        self.emb_dim = args.emb_dim
        self.reg_lambda = args.reg_lambda
        self.reg_type = args.reg_type
        self.loss_type = args.loss_type
        self.device = device
        self.hinge_margin = args.hinge_margin
        self.learning_rel = args.learning_rel
        # ent_h embs is the real part of the embedding and ent_t embs is the imaginary part of the embedding
        self.ent_h_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
        self.ent_t_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
        # rel_embs is the real part of the embedding and rel_inv_embs is the imaginary part of the embedding
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
            self.ent_h_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
            self.ent_t_embs   = nn.Embedding(self.num_ent, args.emb_dim).to(device)
            self.rel_embs     = nn.Embedding(self.num_rel, args.emb_dim).to(device)
            self.rel_inv_embs = nn.Embedding(self.num_rel, args.emb_dim).to(device)

            sqrt_size = 6.0 / np.sqrt(self.emb_dim)
            nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
            nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
            nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
            nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

        elif args.init_type == 'normal':
            for w in weights:
                nn.init.normal_(w, std=args.init_scale)


    def forward(self, heads, rels, tails):
        # real of head
        h_real = self.ent_h_embs(heads)
        # real of tail
        t_real = self.ent_h_embs(tails)
        #img of head
        h_img = self.ent_t_embs(heads)
        #img of tail
        t_img = self.ent_t_embs(tails)
        r_real = self.rel_embs(rels)
        r_img = self.rel_inv_embs(rels)

        if self.learning_rel == 'freeze':
            torch.sum(
                h_real * t_real + h_img * t_img 
                + h_real * t_img - h_img * t_real, dim=1)
        else:
            return torch.sum(
                r_real * h_real * t_real + r_real * h_img * t_img 
                + r_img * h_real * t_img - r_img * h_img * t_real, dim=1)

    def loss(self, score, labels):

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