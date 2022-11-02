import numpy as np
import os, sys, torch, gpytorch, argparse, math, subprocess, re, pickle, yaml
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tune_name', default="gausstypereg", type=str, help='gausstypereg or tilt for now')
    parser.add_argument('-fold', default=0, type=int, help='fold_num')
    args = parser.parse_args()
    return args

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

folders = sorted(os.listdir(path), key=natural_key)
folders = [f for f in folders if 'train' in f]

if __name__ == '__main__':
    args = get_args_review()
    tune_name = args.tune_name
    fold_num= args.fold_num
    cv_type = 'train'
    path = os.path.join('results', tune_name, cv_type, 'fold_{}'.format(fold_num))

    stop = []
    emb_dim = []

    for i in range(len(folders)):
        stop_path = os.path.join(path, 'train_{}'.format(i), 'stop_metric.npy')
        hits = np.load(stop_path)
        stop.append(hits)
        yaml_path = os.path.join(path, 'train_{}'.format( i), 'info.yml')

        with open(os.path.join(path, 'train_{}'.format( i), 'info.yml'), 'r') as f:
            yml = yaml.safe_load(f)
            emb_dim.append(yml["emb_dim"])


    for i in range (len(stop)):
        print("model:",i)
        print('stop metric:',stop[i])
        print('emb_dim:', emb_dim[i])

                


