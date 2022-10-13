from os import X_OK
import sys, torch

def stack_(x):
    x_ = torch.stack((x[0], x[1]))
    s = x_.shape
    return torch.reshape(x_, (s[1], s[0], s[2]))

# INFO: this is gaussian
class UpdateInfo:
    def __init__(self, user_emb, etta, crit_args, model_args, device, crit_rel_emb=None, likes_emb=None):
        self.etta = etta
        self.crit_rel_emb_f = None
        self.crit_rel_emb_inv = None
        
        self.d_f = None
        self.d_inv = None
        self.n = None # update amount, ie. size of last d that was appended 

        if likes_emb is not None:
            #self.likes_emb_f = torch.unsqueeze(likes_emb[0], axis=1)
            #self.likes_emb_inv = torch.unsqueeze(likes_emb[1], axis=1)
            self.likes_emb_f = likes_emb[0]
            self.likes_emb_inv = likes_emb[1]

        self.user_emb_f = user_emb[0]
        self.user_emb_inv = user_emb[1]

        # p(u)
        prec = crit_args.user_prec * torch.eye(model_args.emb_dim).to(device)
        self.user_prec_f = torch.unsqueeze(prec, axis=0) 
        self.user_prec_inv = torch.unsqueeze(prec, axis=0)
        
        # p(d | u)
        self.likelihood_prec = crit_args.default_prec * torch.eye(model_args.emb_dim).to(device)
        self.z_mean = torch.zeros(model_args.emb_dim).to(device)
        self.z_prec = crit_args.z_prec * torch.eye(model_args.emb_dim).to(device)

    # get last element (these are being stored and saved for tracking)
    def get_sampleinfo(self):
        return (torch.mean(self.d_f[-self.n:], axis=0), torch.mean(self.d_inv[-self.n:], axis=0)), self.likelihood_prec, self.n

    def get_priorinfo(self):
        return (self.user_emb_f[-1], self.user_emb_inv[-1]), (self.user_prec_f[-1], self.user_prec_inv[-1])

    # TODO: stack rel
    # store either the user or feeback embs
    def store(self, user_emb=None, d=None, user_prec=None, crit_rel_emb=None):
        if user_emb is not None:
            self.user_emb_f = torch.cat((self.user_emb_f, torch.unsqueeze(user_emb[0], axis=0)))
            self.user_emb_inv = torch.cat((self.user_emb_inv, torch.unsqueeze(user_emb[1], axis=0)))

        if user_prec is not None:
            self.user_prec_f = torch.cat((self.user_prec_f, torch.unsqueeze(user_prec[0], axis=0)))
            self.user_prec_inv = torch.cat((self.user_prec_inv, torch.unsqueeze(user_prec[1], axis=0)))

        if d is not None:
            if self.d_f is None: 
                self.d_f = d[0]
                self.d_inv = d[1] 
            else:                
                self.d_f = torch.cat((self.d_f, d[0]))
                self.d_inv = torch.cat((self.d_inv, d[1]))
            self.n = d[0].shape[0]
   
        if crit_rel_emb is not None:
            if self.crit_rel_emb_f is None:
                self.crit_rel_emb_f = crit_rel_emb[0]
                self.crit_rel_emb_inv = crit_rel_emb[1]
            else:
                torch.cat((self.crit_rel_emb_f, crit_rel_emb[0]))
                torch.cat((self.crit_rel_emb_inv, crit_rel_emb[1]))