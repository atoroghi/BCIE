import os, sys, torch, gpytorch, argparse, math, subprocess, re
import numpy as np
import matplotlib.pyplot as plt

# normalize data to be ~ N(0,1)
def normalize(x):
    return (x - torch.mean(x)) / torch.std(x), torch.mean(x).item(), torch.std(x).item()

# main gp class
class GP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(GP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# convert from continuous data in [0, 1] (gp)
# to parameter for nn training
def normal2param(minmax, x, dtype, base=None):
    assert x >= 0 and x <= 1

    param = (minmax[1] - minmax[0]) * x + minmax[0]
    if base is not None:
        param = math.pow(base, param)

    # return proper type
    if dtype == int: 
        # return x (gp input) corresponding to discretized parameter
        param = int(param)
        if base is not None:
            x_out = math.log(param, base)
        else:
            x_out = param    
        x_out = (x_out - minmax[0]) / (minmax[1] - minmax[0])
        return param, x_out

    elif dtype == float: 
        return param, x
    else:
        print('not implimented yet')
        sys.exit()

def train_sample_gp(x_test, x_train, y_train, batch, dim, n_graph):
    gp_epochs = 100
    
    # gp model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #likelihood.noise = 5e-3
    
    y_train, mean, std = normalize(y_train)
    model = GP(x_train, y_train, likelihood)
    #model.covar.base_kernel.lengthscale = 0.75 / np.power(x_train.shape[0], 1/dim)

    # set fixed noise and lengthscale
    params = set(model.parameters())
    final_params = list(params)
    #    - {model.covar.base_kernel.raw_lengthscale}
    #    - {likelihood.noise_covar.raw_noise})
    optimizer = torch.optim.Adam(params, lr=0.1)  # Includes GaussianLikelihood parameters

    # loss function (marginal log-likelihood) 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # optimization setup
    model.train()
    likelihood.train()

    # train loop
    for e in range(gp_epochs):
        optimizer.zero_grad()

        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()

        optimizer.step()

    # get new points
    model.eval()
    likelihood.eval()

    # get predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if dim != 1:
            x_sample = torch.empty((batch, dim))
        else:
            x_sample = torch.empty(batch)
        
        pred = likelihood(model(x_test))
        for i in range(batch):
            sample = pred.sample().numpy()
            x_sample[i] = x_test[np.argmax(sample)]

        # plot
        if dim == 1:
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111)

            lower, upper = pred.confidence_region()
            ax.plot(x_test.numpy(), pred.mean.numpy(), 'b')
            ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            
            ax.scatter(x_train.numpy(), y_train.numpy(), color='k', s=30)
            ax.scatter(x_sample, np.zeros(batch), color='lime', s=30)

            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.savefig('gp/gp {}.png'.format(n_graph))

    return x_sample