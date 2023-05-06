import math
import torch
import numpy as np
import cvxpy as cp
from scipy.special import expit
import scipy.linalg as sp
import sys

# fast gaussian update
def beta_update(update_info, sn, crit_args, model_args, device, update_type, map_finder, etta, alpha):
    if update_type == "gauss":
        #print(n, "this should be one for direct single")
        (f, inv), prec, n = update_info.get_sampleinfo()
        (f0, inv0), (prec_f0, prec_inv0) = update_info.get_priorinfo()

        # update forward and backward, new priors for user
        out_f = torch.inverse(prec_f0 + n*prec) @ (prec_f0@f0 + n*prec@f)
        out_inv = torch.inverse(prec_inv0 + n*prec) @ (prec_inv0@inv0 + n*prec@inv)
        out_prec_f = prec_f0 + n*prec
        out_prec_inv = prec_inv0 + n*prec
        #print('difference: {}'.format(torch.linalg.norm(f0 - out_f)))

        # store new user prior
        update_info.store(user_emb=(out_f, out_inv), user_prec=(out_prec_f, out_prec_inv))

    elif update_type == "laplace":
        (X_f, X_inv), prec, n = update_info.get_sampleinfo()
        (mu_prior_f, mu_prior_inv), (tau_prior_f, tau_prior_inv) = update_info.get_priorinfo()

        etta_sn = etta[sn]
        #etta_sn = 100

        # in general, the updater is written in the matrix form
        if len(X_f.shape) == 1:
            X_f = torch.unsqueeze(X_f , dim = 0)
            X_inv = torch.unsqueeze(X_inv , dim = 0)

        # finding the new user belief mean
        # Using convext solver to find the W_MAP
        if map_finder == 'cvx':
            try:
                W_new_f = SDR_cvxopt(tau_prior_f, X_f, mu_prior_f, model_args.emb_dim, etta_sn)
                W_new_inv = SDR_cvxopt(tau_prior_inv, X_inv, mu_prior_inv, model_args.emb_dim, etta_sn)
                W_new_f = torch.tensor(W_new_f).float().to(device)
                W_new_inv = torch.tensor(W_new_inv).float().to(device)
            except:
                W_new_f = GDOPT(tau_prior_f, X_f, mu_prior_f, etta_sn, alpha)
                W_new_inv = GDOPT(tau_prior_inv, X_inv, mu_prior_inv, etta_sn, alpha)
        elif map_finder == 'gd':
            W_new_f = GDOPT(tau_prior_f, X_f, mu_prior_f, etta_sn, alpha)
            W_new_inv = GDOPT(tau_prior_inv, X_inv, mu_prior_inv, etta_sn, alpha)

        # finding the new user belief precision 
            # first, calculate Hessian of the log likelihood
        
        
        _, H_map_f = log_likelihood(X_f, W_new_f, etta_sn)
        _, H_map_inv = log_likelihood(X_inv, W_new_inv, etta_sn)

        #making sure the returned precision is symmetric

        za_f = tau_prior_f + etta_sn * H_map_f
        za_inv = tau_prior_inv + etta_sn * H_map_inv

        H_out_f = torch.maximum(za_f,za_f.T)
        H_out_inv = torch.maximum(za_inv,za_inv.T)

        update_info.store(user_emb=(W_new_f, W_new_inv), user_prec=(H_out_f, H_out_inv))

def beta_update_indirect(update_info, sn, crit_args, model_args, device, update_type, map_finder, etta, alpha):
    if update_type == "gauss":
        (user_mean_f, user_mean_inv), (user_prec_f, user_prec_inv) = update_info.get_priorinfo()
        (likes_emb_f, likes_emb_inv) = (update_info.likes_emb_f , update_info.likes_emb_inv)
        (evidence_f, evidence_inv), prec_evidence, _ = update_info.get_sampleinfo()

        (item_mean_f, item_mean_inv) = (update_info.z_mean, update_info.z_mean)
        (item_prec_f, item_prec_inv) = (update_info.z_prec, update_info.z_prec)


        # get information matrix
        h_u_f = user_prec_f @ user_mean_f
        h_u_inv = user_prec_inv @ user_mean_inv

        # get factor between user and z (likes relation)
        D_r2 = torch.diag(likes_emb_f)
        D_r2_inv = torch.diag(likes_emb_inv)

        # update z given evidence d
        #item_mean_f = evidence_f
        #item_mean_inv = evidence_inv
        n = 1 
        # update mean of z given evidence d
        temp_f = (item_prec_f @ item_mean_f + n * prec_evidence @ evidence_f)
        mu_z_f = torch.inverse(item_prec_f + n * prec_evidence) @ temp_f 

        temp_inv = (item_prec_inv @ item_mean_inv + n * prec_evidence @ evidence_inv)
        mu_z_inv = torch.inverse(item_prec_inv + n * prec_evidence) @ temp_inv

        # update prec of z given evidence d
        J_z_f = item_prec_f + n * prec_evidence
        J_z_inv = item_prec_inv + n * prec_evidence

        # p(u|z)
        temp_f_2 = (user_prec_f @ user_mean_f + n * J_z_f @ mu_z_f)
        user_mean_updated_f= torch.inverse(user_prec_f + n * J_z_f) @ temp_f_2 
        temp_inv = (user_prec_inv @ user_mean_inv + n * J_z_inv @ mu_z_inv)
        user_mean_updated_inv = torch.inverse(user_prec_inv + n * J_z_inv) @ temp_inv
        user_prec_updated_f = user_prec_f + n * J_z_f
        user_prec_updated_inv = user_prec_inv + n * J_z_inv

        #joint approach
        #J_z_f_inv = torch.inverse(J_z_f)
        #J_z_inv_inv = torch.inverse(J_z_inv)

        # potential of user updated (information form)
        #h_u_updated_f = h_u_f - D_r2 @ mu_z_f
        #h_u_updated_inv = h_u_inv - D_r2_inv @ mu_z_inv

        # precision of user updated
        #user_prec_updated_f = user_prec_f - D_r2 @ J_z_f_inv @ D_r2
        #user_prec_updated_inv = user_prec_inv - D_r2_inv @ J_z_inv_inv @ D_r2_inv

        # convert from information to mean prec form
        #user_mean_updated_f = torch.inverse(user_prec_updated_f) @ h_u_updated_f
        #user_mean_updated_inv = torch.inverse(user_prec_updated_inv) @ h_u_updated_inv

        update_info.store(user_emb=(user_mean_updated_f, user_mean_updated_inv), user_prec=(user_prec_updated_f, user_prec_updated_inv))
    
    if update_type == "laplace":
        (X_f, X_inv), prec, n = update_info.get_sampleinfo()
        (mu_prior_f, mu_prior_inv), (tau_prior_f, tau_prior_inv) = update_info.get_priorinfo()
        (item_mean_f, item_mean_inv) = (update_info.z_mean, update_info.z_mean)
        (item_prec_f, item_prec_inv) = (update_info.z_prec, update_info.z_prec)

        # in general, the updater is written in the matrix form
        if len(X_f.shape) == 1:
            X_f = torch.unsqueeze(X_f , dim = 0)
            X_inv = torch.unsqueeze(X_inv , dim = 0)

        etta_sn = etta[sn]
        if map_finder == 'cvx':
            try:
                z_map_f = SDR_cvxopt(item_prec_f, X_f, item_mean_f, model_args.emb_dim, etta_sn)
                z_map_inv = SDR_cvxopt(item_prec_inv, X_inv, item_mean_inv, model_args.emb_dim, etta_sn)
                z_map_f = torch.tensor(z_map_f).float().to(device)
                z_map_inv = torch.tensor(z_map_inv).float().to(device)
            except:
                print("using GD")
                z_map_f = GDOPT(item_prec_f, X_f, item_mean_f, etta_sn, alpha)
                z_map_inv = GDOPT(item_prec_inv, X_inv, item_mean_inv, etta_sn, alpha)

        elif map_finder == 'gd':
            z_map_f = GDOPT(item_prec_f, X_f, item_mean_f, etta_sn, alpha)
            z_map_inv = GDOPT(item_prec_inv, X_inv, item_mean_inv, etta_sn, alpha)
        X_new_f = update_info.likes_emb_f  * z_map_f
        X_new_inv = update_info.likes_emb_inv * z_map_inv

        if len(X_new_f .shape) == 1:
            X_new_f = torch.unsqueeze(X_new_f , dim = 0)
            X_new_inv = torch.unsqueeze(X_new_inv , dim = 0)

        if map_finder == 'cvx':
            try:
                W_new_f = SDR_cvxopt(tau_prior_f, X_new_f, mu_prior_f, model_args.emb_dim, etta_sn)
                W_new_inv = SDR_cvxopt(tau_prior_inv, X_new_inv, mu_prior_inv, model_args.emb_dim, etta_sn)
                W_new_f = torch.tensor(W_new_f).float().to(device)
                W_new_inv = torch.tensor(W_new_inv).float().to(device)
            except:
                W_new_f = GDOPT(tau_prior_f, X_new_f, mu_prior_f, etta_sn, alpha)
                W_new_inv = GDOPT(tau_prior_inv, X_new_inv, mu_prior_inv, etta_sn,alpha)
        elif map_finder == 'gd':
            W_new_f = GDOPT(tau_prior_f, X_new_f, mu_prior_f, etta_sn, alpha)
            W_new_inv = GDOPT(tau_prior_inv, X_new_inv, mu_prior_inv, etta_sn,alpha)

        _, H_map_f = log_likelihood(X_new_f, W_new_f, etta_sn)
        _, H_map_inv = log_likelihood(X_new_inv, W_new_inv, etta_sn)
        za_f = tau_prior_f + etta_sn * H_map_f
        za_inv = tau_prior_inv + etta_sn * H_map_inv
        H_out_f = torch.maximum(za_f,za_f.T)
        H_out_inv = torch.maximum(za_inv,za_inv.T)
        update_info.store(user_emb=(W_new_f, W_new_inv), user_prec=(H_out_f, H_out_inv))


def SDR_cvxopt(landa, X_all , previous_w, emb_dim, etta):
    w = cp.Variable(emb_dim)
    previous_w = previous_w.cpu()
    landa = landa.cpu()
    X_all = X_all.cpu()
    objective_function = 0.5 * cp.quad_form(w-previous_w, landa)

    for i in range(len(X_all)):
        var= (X_all[i] @ w)
        objective_function += etta * cp.logistic(-1 * var)

    prob2 = cp.Problem(cp.Minimize(objective_function))
    prob2.solve()

    return w.value

def log_likelihood(X, W, etta):
    N = (X.shape)[0]
    
    #logits = torch.mv(X, w)
    logits = X@ W

    probs = torch.clip(torch.sigmoid(logits * etta), 1e-20, 1-1e-20)

    #probs = np.clip(expit(logits * etta), 1e-20, 1-1e-20)

    #g = torch.mv(X.t(), (probs - 1))

    g = X.t() @ (probs - 1)
    H = X.t() @ torch.diag(((probs*(1 - probs)))) @ X

    #H = np.matmul(np.matmul(np.transpose(X),np.diag((probs*(1 - probs)))), X)
    return g,H
    

# performing gradient descent for laplace approximation    

def GDOPT(tau_prior, X, W,  etta, alpha):

    W_last = 1*W 
    W_updated = torch.zeros_like(W)
    
    prev_obj = 10
    curr_obj = 0

    # while the objective function is still changing, perform gradient descent
    while abs(prev_obj - curr_obj) > 0.00001:
        #previous objective set to current objective
        prev_obj = 1 * curr_obj

        # Get negative of gradients of the likelihood and prior
        neg_g_likelihood , _ = log_likelihood(X, W, etta)
        neg_g_prior = tau_prior @ (W_updated - W_last)

        # Calculate the posterior gradient
        neg_g = neg_g_prior + etta * neg_g_likelihood

        # Gradient descent and recalculation of the objective value
        W_updated  = W_updated - alpha * neg_g
        curr_obj = 0.5 * (W_updated -W_last).t() @ tau_prior @ (W_updated -W_last) - etta * torch.log(torch.sigmoid(X@ W_updated ))

    return W_updated











class Updater:
    def __init__(self, X, y, mu_prior, tau_prior, crit_args, model_args, device, etta):
        self.X = X
        self.y = y
        self.W = mu_prior
        # this is precision not variance
        self.tau_prior = tau_prior
        self.W = mu_prior
        self.alpha = crit_args.alpha
        #self.max_iters= args.max_iters_laplace
        self.etta = etta
        self.emb_dim= model_args.emb_dim
        self.device = device
        self.update_type = crit_args.update_type
        self.likelihood_precision = crit_args.likelihood_precision

# The main updating function that performs gaussian or laplace updating
    def compute_laplace_approximation(self):
        assert self.update_type in ['gaussian', 'laplace']
        if self.update_type == "gaussian":
            # Update formula https://en.wikipedia.org/wiki/Conjugate_prior
            n = self.X.shape[0]
            tau_likelihood= self.likelihood_precision * np.eye(self.emb_dim)
            za = self.tau_prior + n * tau_likelihood
            H_out = np.maximum(za,za.T)
            mu = np.transpose(np.matmul(np.linalg.inv(H_out) , np.transpose(((np.matmul(self.W, self.tau_prior)) + np.matmul(n *self.W, tau_likelihood )))))

        if self.update_type == "laplace":
            # derivation : https://www.overleaf.com/read/jypckmmmcvsv
            #solving convex problem to update user belief

            W_new = self.SDR_cvxopt(self.tau_prior, self.X, self.y , self.W)
            self.W = W_new
            # log likelihood for Hessian update
            _, H_map = self.log_likelihood()

            prior_precision = self.tau_prior
            # Hessian update (prior precision + log likelihood)
            za = prior_precision + self.etta*H_map
            H_out = np.maximum(za,za.T)

        return self.W, H_out
    # using the convex solver to update user belief
    def SDR_cvxopt(self,landa, X_all, y , previous_w):
        w = cp.Variable(self.emb_dim)
        #constraints = [cp.norm(w) <= 100*np.sqrt(128)]
        previous_w = np.reshape(previous_w, self.emb_dim)
        objective_function = cp.quad_form(w-previous_w, landa)
        for i in range(len(X_all)):
            var= (X_all[i] @ w) * y[i]
            objective_function += self.etta * cp.logistic(-1 * var)
        #prob = cp.Problem(cp.Minimize(objective_function), constraints)
        #prob2 = cp.Problem(cp.Minimize(objective_function),constraints)
        prob2 = cp.Problem(cp.Minimize(objective_function))

        #try:
        #	prob.solve(solver=cp.CVXOPT)
        #except: 
        #print("etta:",self.etta)
        #print("w_prev",previous_w)
        #print("X_all",X_all)
        #print("landa",landa)
        #prob2.solve(verbose=True)
        prob2.solve()
        #print(prob2.status)
       

        return w.value
# Calculating the Hessian of log likelihood
    def log_likelihood(self):
        N = np.shape(self.X)[0]
        logits = np.matmul(self.X, self.W)
        probs = np.clip(expit(self.y * logits * self.etta), 1e-20, 1-1e-20)
        H = np.matmul(np.matmul(np.transpose(self.X),np.diag((probs*(1 - probs)))), self.X)

        return H

    #def gradient_descent(self, g):
        #self.W = self.W - self.alpha * g
        #return self.W

    #def log_prior(self):
#		nlp = - self.prior_distribution.log_prob(self.W)
#		#g: gradient of negative log prior
#		g = (self.W-self.mu_prior) @ self.lam
#		return nlp, g
