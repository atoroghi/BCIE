import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
plt.style.use('seaborn')

folder = ['softplus', 'single', 'gauss', 'svd', 'rev_double', 'pseudo_svd']
leg = ['split-reverse', 'joined', 'gauss', 'svd', 'split-regular', 'simple-like-svd']
groups = ['hit@1', 'hit@3', 'hit@10'] 
color = ['r', 'k', 'b', 'm', 'g']

all_hits, all_err = [], []
for j, f in enumerate(folder):
    sub_hits, sub_err = [], []
    for i in range(3):
        path = os.path.join('results', f, 'results.txt')

        hits = np.genfromtxt(path, delimiter=' ')
        n = hits.shape[0]
        d = hits[:, i]
        (_ , up) = st.t.interval(alpha=0.95, df=n-1, loc=np.mean(d), scale=st.sem(d))
        err = up - np.mean(d)
        
        sub_err.append(err)
        sub_hits.append(np.mean(d))

    all_err.append(sub_err)
    all_hits.append(sub_hits)
    
val_dic, err_dic = {}, {}
for i, h in enumerate(all_hits):
    print(leg[i])
    val_dic.update({leg[i] : h})

for i, e in enumerate(all_err):
    err_dic.update({leg[i] : e})

print(val_dic, err_dic)

vdf = pd.DataFrame(val_dic, index=groups)
edf = pd.DataFrame(err_dic, index=groups)

vdf.plot(
    kind='bar', yerr=edf, figsize=(12,10), rot=0, 
    title='Confidence Intervals - 95%', ylabel='Hit Percentage'
)
#plt.show()
plt.savefig(os.path.join('results', 'intervals.jpg'))
plt.close()




#plt.figure(figsize=(12,10))

#plt.errorbar(i, np.mean(d), yerr=err, color=color[j], capsize=4)
#plt.scatter(i, np.mean(d), color=color[j], marker='_')

#plt.legend(leg)
#plt.xticks(ticks=[-1, 0, 1, 2, 3], labels=[' ', , ' '])
#plt.title('Confidence Intervals - 95%')
#plt.ylabel('Hit Percentage')
#plt.show()