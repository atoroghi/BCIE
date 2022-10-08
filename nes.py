#%%
#%%
import numpy as np
import os
import pickle
main_path = os.path.join('datasets', 'ML_FB')
kg = np.load(os.path.join('datasets', 'ML_FB', 'kg.npy'), allow_pickle=True)        
rels,counts=np.unique(kg[:,1], return_counts=True)
print(counts)
# %%
