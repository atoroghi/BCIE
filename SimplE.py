import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SimplE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, reg_lambda, device):
        super(SimplE, self).__init__()
        self.num_ent = num_ent + 1
        self.num_rel = num_rel + 1
        self.emb_dim = emb_dim
        self.reg_lambda = reg_lambda
        self.device = device

        self.ent_h_embs   = nn.Embedding(self.num_ent, emb_dim).to(device)
        self.ent_t_embs   = nn.Embedding(self.num_ent, emb_dim).to(device)
        self.rel_embs     = nn.Embedding(self.num_rel, emb_dim).to(device)
        self.rel_inv_embs = nn.Embedding(self.num_rel, emb_dim).to(device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    def forward(self, x):
        hh_embs = self.ent_h_embs(x[:,:,0])
        ht_embs = self.ent_h_embs(x[:,:,2])
        th_embs = self.ent_t_embs(x[:,:,0])
        tt_embs = self.ent_t_embs(x[:,:,2])
        r_embs = self.rel_embs(x[:,:,1])
        r_inv_embs = self.rel_inv_embs(x[:,:,1])

        # get forward and inverse similarity
        for_sim = hh_embs * r_embs * tt_embs
        inv_sim = ht_embs * r_inv_embs * th_embs

        return torch.clamp((for_sim + inv_sim) / 2, -20, 20)

    def loss(self, score, x):
        labels = torch.unsqueeze(x[:,:,3], axis=2)
        loss = torch.sum(F.softplus(-labels * score))
        return loss, self.reg_loss()

    def reg_loss(self):
        return self.reg_lambda * (
            (torch.norm(self.ent_h_embs.weight, p=2) ** 2) \
            + (torch.norm(self.ent_t_embs.weight, p=2) ** 2) \
            + (torch.norm(self.rel_embs.weight, p=2) ** 2) \
            + (torch.norm(self.rel_inv_embs.weight, p=2) ** 2)
        )

    #def forward(self, heads, rels, tails):
        #hh_embs = self.ent_h_embs(heads)
        #ht_embs = self.ent_h_embs(tails)
        #th_embs = self.ent_t_embs(heads)
        #tt_embs = self.ent_t_embs(tails)
        #r_embs = self.rel_embs(rels)
        #r_inv_embs = self.rel_inv_embs(rels)

        #scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        #scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        #return torch.clamp((scores1 + scores2) / 2, -20, 20) , hh_embs , r_embs , tt_embs , ht_embs , r_inv_embs , th_embs




       