import math
import torch
import numpy as np
import cvxpy as cp
from scipy.special import expit
import scipy.linalg as sp
import sys

# TODO: no comments or explanation of how this is supposed to work
class Updater:

	def __init__(self, X, y, mu_prior, tau_prior, model_args, crit_args, device, etta):
		self.X = X
		self.y = y
		self.W = mu_prior
		# this is precision not variance
		self.tau_prior = tau_prior
		#elf.alpha= crit_args.alpha
		#self.max_iters= args.max_iters_laplace
		self.etta= etta
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
