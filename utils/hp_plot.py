import os, sys, torch
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

path = 'gp'
hps = [
    'lr', 'batch size', 'emb dim', 'reg lambda', 
    'kg lambda', 'init scale', 'neg ratio', 'neg_power'
]

# look at all exps
for i, f in enumerate(os.listdir(path)):
    exp_path = os.path.join(path, f)
    print(f)

    # for each fold
    for j, g in enumerate(os.listdir(exp_path)):
        load_path = os.path.join(exp_path, g)
        x = torch.load(os.path.join(load_path, 'x_train.pt'))
        y = torch.load(os.path.join(load_path, 'y_train.pt'))

        x = x.numpy()
        y = y.numpy()

        # make plots
        fig = plt.figure(figsize=(12,6))

        for k in range(x.shape[1]):
            s = int('2' + '4' + str(k+1))
            ax = fig.add_subplot(s)

            ax.scatter(x[:,k], y)
            ax.set_title('{}: {}'.format(f, hps[k]))
        plt.tight_layout()
        plt.savefig(os.path.join(load_path, 'hp_plot.jpg'))
        plt.close()
        #plt.show()
        break