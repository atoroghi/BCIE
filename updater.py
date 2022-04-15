import math
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import cvxpy as cp
seed = torch.manual_seed(1)

class Updater:
	def __init__(self,X,y,mu_prior,Sigma_prior,W,args,etta):
		self.X = X
		self.y = y
		self.mu_prior = mu_prior
		self.Sigma_prior = Sigma_prior
		self.lam= torch.inverse(Sigma_prior)
		self.W= W
		self.prior_distribution=MultivariateNormal(mu_prior, covariance_matrix=Sigma_prior)
		self.alpha=args.alpha
		self.max_iters=args.max_iters_laplace
		self.etta=etta
		self.emb_dim=args.emb_dim
		self.device = device

	def gradient_descent(self, g):
		self.W = self.W - self.alpha * g

		return self.W

	def SDR_cvxopt(self,Sigma_prior, X_all, y , previous_w):
		X_all = X_all.cpu()
		y = y.cpu()
		landa = torch.inverse(Sigma_prior).cpu()
		previous_w=previous_w.cpu().detach().numpy()
		w= cp.Variable(self.emb_dim)
		constraints=[]
		objective_function= 0.5*cp.quad_form(w-previous_w, landa)
		for i in range(len(X_all)):
			var= (X_all[i]@w)*y[i]
			objective_function += cp.logistic(-1*self.etta*var)
		prob = cp.Problem(cp.Minimize(objective_function), constraints)
		prob.solve()
		return w.value , prob.value

	def log_prior(self):
		nlp = - self.prior_distribution.log_prob(self.W)
		#g: gradient of negative log prior
		g = (self.W-self.mu_prior) @ self.lam
		return nlp, g

	def log_likelihood(self):
		(N,_) = self.X.size()
		logits = torch.mv(self.X, self.W)
		probs1 = torch.sigmoid(-1*self.y*logits*self.etta).clamp(min=1e-6, max=1-1e-6)
		probs = torch.sigmoid(self.y*logits*self.etta).clamp(min=1e-6, max=1-1e-6)
		#nll= 1 * torch.sum(torch.log(probs1))
		nll= -1 * torch.sum(torch.log(probs))
		#g: gradient of negative log likelihood
		#g= torch.mv(self.X.t(),self.y*probs)
		g= -1*torch.mv(self.X.t(),self.y*probs1)
		H = torch.mm(torch.mm(self.X.t(), torch.diag((probs * (1 - probs)))), self.X)
		return nll, g, H



	def compute_laplace_approximation(self):
		#print("X"+str(self.X))
		#print("y"+str(self.y))
		#W_old=100
		#i=0
		#while i<self.max_iters and torch.sum(abs(self.W-W_old))>0.0001:
		#	W_old= self.W
		#	nll, g_nll, H_nll = self.log_likelihood()
		#	nlp, g_prior = self.log_prior()
		#	g = g_nll + g_prior
		#	i=i+1
		#	self.W = self.gradient_descent(g)
		W_new, _ = self.SDR_cvxopt(self.Sigma_prior, self.X, self.y , self.W)
		self.W = W_new
		mu = self.W
		_, _, H_map = self.log_likelihood()
		prior_precision = self.lam
		Sigma = torch.inverse(prior_precision + self.etta*H_map)
		return mu, Sigma
