import math
import torch
import numpy as np
import cvxpy as cp
from scipy.special import expit
import scipy.linalg as sp

# TODO: no comments or explanation of how this is supposed to work
class Updater:
	# TODO: too many parameters, this should just accept args and unpack it in the class init
	def __init__(self, X, y, mu_prior, Sigma_prior, W, args, etta, device, update_type, likelihood_precision):
		self.X = X
		self.y = y
		self.mu_prior = mu_prior

		# this is precision not variance
		# TODO: where is tau_prior from
		self.tau_prior = tau_prior
		self.W = W
		self.alpha=args.alpha
		self.max_iters=args.max_iters_laplace
		self.etta=etta
		self.emb_dim=args.emb_dim
		self.device = device
		self.update_type = args.update_type
		self.likelihood_precision = args.likelihood_precision

	def compute_laplace_approximation(self):
		# TODO where is args from
		assert args.update_type in ['gaussian', 'laplace']
		if self.update_type == "gaussian":
			n = self.X.size()[0]
			H_out = self.prior_precision + n * self.likelihood_precision
			mu = (self.prior_precision * self.mu_prior + self.likelihood_precision * self.X.sum()) * torch.cholesky_inverse(H_out)

		if self.update_type == "laplace":
			W_new = self.SDR_cvxopt(self.tau_prior, self.X, self.y , self.W)
			self.W = W_new
			mu = self.W
			H_map = self.log_likelihood()

			prior_precision = self.tau_prior
			za = prior_precision + self.etta*H_map
			H_out = np.maximum(za,za.T)

		return mu, H_out

	def SDR_cvxopt(self,landa, X_all, y , previous_w):
		w = cp.Variable(self.emb_dim)
		constraints = []
		objective_function = cp.quad_form(w-previous_w, landa)
		for i in range(len(X_all)):
			var= (X_all[i] @ w) * y[i]
			objective_function += self.etta * cp.logistic(-1 * var)
		prob = cp.Problem(cp.Minimize(objective_function), constraints)
		prob2 = cp.Problem(cp.Minimize(1000*objective_function),constraints)

		# TODO: this is bad 
		try: prob.solve()
		except: prob2.solve()
		return w.value

	def log_likelihood(self):
		N = np.shape(self.X)[0]
		logits = np.matmul(self.X, self.W)
		probs = np.clip(expit(self.y * logits * self.etta), 1e-6, 1-1e-6)
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
