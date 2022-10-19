import numpy as np
import os, sys, torch, gpytorch, argparse, math, subprocess, re, pickle, yaml


tune_name = 'gausstypereg'
cv_type = 'train'
fold_num = 0
path = os.path.join('results', tune_name, cv_type, 'fold_{}'.format(fold_num))

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

folders = sorted(os.listdir(path), key=natural_key)
folders = [f for f in folders if 'train' in f]


stop = []
emb_dim = []

for i in range(len(folders)):
    stop_path = os.path.join(path, 'train_{}'.format(len(folders) + i), 'stop_metric.npy')
    hits = np.load(stop_path)
    stop.append(hits)
    yaml_path = os.path.join(path, 'train_{}'.format(len(folders) + i), 'info.yml')

    with open(os.path.join(path, 'train_{}'.format(len(folders) + i), 'info.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        emb_dim.append(yml["emb_dim"])

for i in range (len(stop)):

    print('stop metric:',stop[i])
    print('emb_dim:', emb_dim[i])

                


