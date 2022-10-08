import math
import torch
import numpy as np
import cvxpy as cp
from scipy.special import expit
import scipy.linalg as sp
import sys

# fast gaussian update
def beta_update(update_info, sn, crit_args, model_args, device):
	n = update_info.d_f.shape[0] # number of update samples
	(f, inv), prec = update_info.get_sampleinfo()
	(f0, inv0), (prec_f0, prec_inv0) = update_info.get_priorinfo()

	# update forward and backward, new priors for user
	out_f = torch.inverse(prec_f0 + n*prec) @ (prec_f0@f0 + n*prec@f)
	out_inv = torch.inverse(prec_inv0 + n*prec) @ (prec_inv0@inv0 + n*prec@inv)
	out_prec_f = prec_f0 + n*prec
	out_prec_inv = prec_inv0 + n*prec

	# store new user prior
	update_info.store(user_emb=(out_f, out_inv), user_prec=(out_prec_f, out_prec_inv))

#TODO: We need rel_emb, rel_emb_inv, 
def beta_update_indirect(update_info, sn, crit_args, model_args, device):
    #(evidence_mean_f, evidence_mean_inv), (evidence_prec_f, evidence_prec_inv) = update_info.get_mean_prec()
    (user_mean_f, user_mean_inv, user_prec_f, user_prec_inv) = (update_info.user_emb_f[0], update_info.user_emb_inv[0], update_info.user_prec, update_info.user_prec)
    (likes_emb_f, likes_emb_inv) = (update_info.likes_emb_f[0] , update_info.likes_emb_inv[0])
    (evidence_f, evidence_inv) = (update_info.d_f, update_info.d_inv)
    (rel_emb_f, rel_emb_inv) = (update_info.crit_rel_emb_f, update_info.crit_rel_emb_inv)
    (item_mean_f, item_mean_inv) = (update_info.z_mean, update_info.z_mean)
    (item_prec_f, item_prec_inv) = (update_info.z_prec, update_info.z_prec)

    h_u_f = user_prec_f.cuda()
    D_r1 = torch.diag(rel_emb_f)
    D_r2 = torch.diag(likes_emb_f)
    J_z_inv = torch.inverse(item_prec_f)
    h_z_f = item_prec_f @ item_mean_f
    h_u_updated_f = h_u_f - 0.5*D_r2 @ J_z_inv @ (h_z_f + D_r1 @evidence_f)
    user_prec_updated_f = user_prec_f - D_r1 @ J_z_inv @ D_r1
    user_mean_updated_f = torch.inverse(user_prec_updated_f) @ h_u_updated_f
    return user_prec_updated_f, user_mean_updated_f

# TODO: no comments or explanation of how this is supposed to work
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
			H_map = self.log_likelihood()

			prior_precision = self.tau_prior
            # Hessian update (prior precision + log likelihood)
			za = prior_precision + self.etta*H_map
			H_out = np.maximum(za,za.T)

		return self.W, H_out
    # using the convex solver to update user belief
	def SDR_cvxopt(self,landa, X_all, y , previous_w):
		w = cp.Variable(self.emb_dim)
		constraints = [cp.norm(w) <= 100*np.sqrt(128)]
		previous_w = np.reshape(previous_w, self.emb_dim)
		objective_function = cp.quad_form(w-previous_w, landa)
		for i in range(len(X_all)):
			var= (X_all[i] @ w) * y[i]
			objective_function += self.etta * cp.logistic(-1 * var)
		#prob = cp.Problem(cp.Minimize(objective_function), constraints)
		prob2 = cp.Problem(cp.Minimize(objective_function),constraints)

		# TODO: this is bad 
        # armin: this is bad, but it's because cvxpy is unstable with small 
        # numbers and you have to scale the objective function if it fails. Clearly, the solution 
        # doesn't change though. We should probably try more advanced solvers such as MOSEK
        # http://ask.cvxr.com/t/scaling-and-numerical-stability/320/3
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
