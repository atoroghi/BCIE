import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

folder = ['softplus', 'gauss']
leg = ['double', 'gauss']
color = ['k', 'r']

for i in range(3):
    for j, f in enumerate(folder):
        path = os.path.join('results', f, 'results.txt')

        hits = np.genfromtxt(path, delimiter=' ')
        n = hits.shape[0]
        d = hits[:, i]
        (_ , up) = st.t.interval(alpha=0.95, df=n-1, loc=np.mean(d), scale=st.sem(d))
        err = up - np.mean(d)
        print(np.mean(d))

        plt.errorbar(i, np.mean(d), yerr=err, color=color[j], capsize=4)
        #plt.scatter(i, np.mean(d), color=color[j], marker='_')

plt.legend(leg)
plt.xticks(ticks=[-1, 0, 1, 2, 3], labels=[' ', 'hit@1', 'hit@3', 'hit@10', ' '])
plt.title('Confidence Intervals - 95%')
plt.ylabel('Hit Percentage')
plt.show()
