from abc import update_abstractmethods
import torch

# we want:
# class that stores all info for updates 
# pass this into updaters 
# general (ie. works with all updaters given upate_type on init)
    # 1) -----use this------
        # parent class: etta, and user info, critiques / feedback
        # child class: all other update parameters
        # pass in update function 
    # 2) 
        # everything in one big class, use if statements

# FOR NOW, SMALL NUM OF USERS
# aes = "at each session"
# gaussian case
    # rank of gt aet
    # all covar / prec, all user vector
        # if too large: det(covar / prec), ||user_n - user_n+1||
    
    # aes: mean and std dev of mrr, det covar, norm diff
    # full histogram aes

# return info about each of the priors
# this class stores all history for saving / tracking 
class UpdateInfo:
    def __init__(self, user, etta, crit_args, model_args):
        # used for update strength
        self.etta = etta

        # assume default for n=1 samples, calculate otherwise
        self.user_f = user[0]
        self.user_inv = user[1]
        self.user_cov_default = crit_args.user_cov * torch.eye(model_args.emb_dim)

        # the model defines this. N(0, lambda*I)
        self.z_f = None
        self.z_inv = None

        # default covar for case when n=1 samples and covar dne
        self.z_cov = torch.eye(model_args.emb_dim) / model_args.reg_lambda

    # get mean and covar for bayesian update
    def mean_covar(self, sn):
        if sn == 0: 
            return (self.z_f, self.z_inv), (self.z_cov, self.z_cov)
        else:
            mean_f = torch.mean(self.z_f)
            mean_inv = torch.mean(self.z_inv)
            covar_f = torch.covar(self.z_f)
            covar_inv = torch.covar(self.z_inv)

            return (mean_f, mean_inv), (covar_f, covar_inv)

    # store all user updates and d feedback 
    def store(self, user=None, z=None):
        # format is [n, embeddings]
        if user is not None:
            self.user_f = torch.cat((self.user_f, user[0]))
            self.user_inv = torch.cat((self.user_inv, user[1]))

        if z is not None:
            if self.z_f is None:
                self.z_f = z[0]
                self.z_inv = z[1]
            else:
                self.z_f = torch.cat((self.z_f, z[0]))
                self.z_inv = torch.cat((self.z_inv, z[1]))
