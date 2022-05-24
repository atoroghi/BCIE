

import math
#import torch
#from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import cvxpy as cp
from requests import session
#seed = torch.manual_seed(1)
from scipy.special import expit
import scipy.linalg as sp

class Updater:
	def __init__(self,X,y,mu_prior,Sigma_prior,W,args,etta,device,session_no):
		#self.X = X.cpu()
		self.X = X
		#self.y = y.cpu()
		self.y = y
		#self.mu_prior = mu_prior.cpu()
		self.mu_prior = mu_prior
		#self.Sigma_prior = Sigma_prior.cpu()
		self.Sigma_prior = Sigma_prior
		#self.lam= torch.cholesky_inverse(Sigma_prior)
		##zz , _ = sp.lapack.dpotrf(Sigma_prior, False, False)
		##inv_M , _ = sp.lapack.dpotri(zz)
		#self.lam = np.linalg.inv(Sigma_prior)
		##self.lam = np.triu(inv_M) + np.triu(inv_M, k=1).T
		#self.W= W.cpu()
		self.W = W
		#self.prior_distribution=MultivariateNormal(mu_prior, covariance_matrix=Sigma_prior)
		##self.prior_distribution = np.random.multivariate_normal(mu_prior,Sigma_prior)
		self.alpha=args.alpha
		self.max_iters=args.max_iters_laplace
		self.etta=etta
		self.emb_dim=args.emb_dim
		self.device = device
		self.session_no = session_no
		self.etta_dict={1:0.00001,2:0.1,3:0.5,4:1,5:10}
		#self.etta_cvx = session_no



	def gradient_descent(self, g):
		self.W = self.W - self.alpha * g

		return self.W

	##def SDR_cvxopt(self,Sigma_prior, X_all, y , previous_w):
	def SDR_cvxopt(self,landa, X_all, y , previous_w):
		#X_all = X_all.cpu()
		#y = y.cpu()
		##landa = np.linalg.inv(Sigma_prior)
		#landa = torch.cholesky_inverse(Sigma_prior).cpu()
		#previous_w=previous_w.cpu().detach().numpy()
		w= cp.Variable(self.emb_dim)
		constraints=[]
		objective_function= cp.quad_form(w-previous_w, landa)
		for i in range(len(X_all)):
			var= (X_all[i]@w)*y[i]
			#objective_function += cp.logistic(-1*self.etta*var)
			objective_function += self.etta_dict[self.session_no]*cp.logistic(-1*var)
			#objective_function += self.etta_cvx*cp.logistic(-1*var)
		prob = cp.Problem(cp.Minimize(objective_function), constraints)
		prob2 = cp.Problem(cp.Minimize(1000*objective_function),constraints)
		try:
			prob.solve()
		except:
			prob2.solve()
		
		return w.value , prob.value

	#def log_prior(self):
#		nlp = - self.prior_distribution.log_prob(self.W)
#		#g: gradient of negative log prior
#		g = (self.W-self.mu_prior) @ self.lam
#		return nlp, g

	def log_likelihood(self):
		#(N,_) = self.X.size()
		N = np.shape(self.X)[0]
		#tensorW=torch.tensor(self.W)
		logits= np.matmul(self.X,self.W)
		#logits = torch.mv(self.X, tensorW.float())
		##probs1 = np.clip(expit(-1*self.y*logits*self.etta),1e-6,1-1e-6)
		#probs1 = torch.sigmoid(-1*self.y*logits*self.etta).clamp(min=1e-6, max=1-1e-6)
		probs = np.clip(expit(self.y*logits*self.etta),1e-6,1-1e-6)
		#probs = torch.sigmoid(self.y*logits*self.etta).clamp(min=1e-6, max=1-1e-6)
		#nll = -1* np.sum(np.log(probs))
		#nll= -1 * torch.sum(torch.log(probs))
		#g: gradient of negative log likelihood
		#g= -1*np.matmul(np.transpose(self.X),self.y*probs1)
		#g= -1*torch.mv(self.X.t(),self.y*probs1)
		H= np.matmul(np.matmul(np.transpose(self.X),np.diag((probs*(1-probs)))),self.X)

		#H = torch.mm(torch.mm(self.X.t(), torch.diag((probs * (1 - probs)))), self.X)
		return H



	def compute_laplace_approximation(self):
		#print("Sigma prior:",self.Sigma_prior)
		#print("mu_input",self.W)

		W_new, _ = self.SDR_cvxopt(self.Sigma_prior, self.X, self.y , self.W)
		self.W = W_new
		mu = self.W
		#print("mu_out:",mu)
		H_map = self.log_likelihood()
		#print("H_map:",H_map)
		prior_precision = self.Sigma_prior
		za = prior_precision + self.etta*H_map
		H_out=np.maximum(za,za.T)
		#print("H_out:",H_out)
		##za , _ = sp.lapack.dpotrf(prior_precision + self.etta*H_map, False, False)
		##inv_A , _ = sp.lapack.dpotri(za)
		##Sigma = np.triu(inv_A) + np.triu(inv_A, k=1).T
		##Sigma = np.linalg.inv(prior_precision + self.etta*H_map)
		return mu, H_out







