from os import X_OK
import sys, torch

def stack_(x):
    x_ = torch.stack((x[0], x[1]))
    s = x_.shape
    return torch.reshape(x_, (s[1], s[0], s[2]))

# TODO: why do we every use tuples for _f vs _inv
class UpdateInfo:
    def __init__(self, user_emb, etta, crit_args, model_args):
        self.etta = etta
        
        self.d_f = None
        self.d_inv = None
        self.user_emb_f = torch.unsqueeze(user_emb[0], axis=0)
        self.user_emb_inv = torch.unsqueeze(user_emb[1], axis=0)

        self.user_prec = crit_args.user_prec * torch.eye(model_args.emb_dim)
        self.default_prec = crit_args.default_prec * torch.eye(model_args.emb_dim)

    # return the sample mean and covariance, if n=1 user the prior prec.
    def get_mean_prec(self):
        if self.d_f.shape[0] > 1: 
            return (self.d_f, self.d_inv), (self.default_prec, self.default_prec)
        else: 
            m_f = torch.mean(self.d_f)
            m_inv = torch.mean(self.d_inv)
            prec_f = torch.inverse(torch.cov(self.d_f))
            prec_inv = torch.inverse(torch.cov(self.d_inv))
            return (m_f, m_inv), (prec_f, prec_inv)

    # store either the user or feeback embs
    def store(self, user_emb=None, d=None):
        if user_emb is not None:
            self.user_emb_f = torch.cat(self.user_emb_f, torch.unsqueeze(user_emb[0], axis=0))
            self.user_emb_inv = torch.cat(self.user_emb_inv, torch.unsqueeze(user_emb[1], axis=0))

        elif d is not None:
            if self.d_f is None: 
                self.d_f = d[0]
                self.d_inv = d[1] 
        else:                
            self.d_f = torch.cat(self.d_f, torch.unsqueeze(d[0], axis=0))
            self.d_inv = torch.cat(self.d_inv, torch.unsqueeze(d[1], axis=0))
   
