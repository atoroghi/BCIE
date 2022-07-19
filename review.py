import os, re, yaml, argparse, torch, math, sys
import numpy as np
import matplotlib.pyplot as plt

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev')
    return parser.parse_args() 

def normal2param(minmax, param, dtype, base=None):
    # return proper type
    if base is not None:
        x_out = math.log(param, base)
    else:
        x_out = param    
    x_out = (x_out - minmax[0]) / (minmax[1] - minmax[0])

    return x_out

def convert(args):
    lr_range = [-2, 1] # 10 ^ lr_range
    batch_range = [11, 14] # 2 ^ batch_range
    emb_range = [3, 8] # 2 ^ emb_range
    reg_range = [-5, 1] # 10 ^ reg_range
    kg_range = [-5, 1] # 10 ^ kg_range
    ratio_range = [1, 15]
    power_range = [0, 1]

    x = np.random.rand(6)
    
    x[0] = normal2param(emb_range, args.emb_dim, int, base=2)
    x[1] = normal2param(batch_range, args.batch_size, int, base=2)
    x[2] = normal2param(lr_range, args.lr, float, base=10)
    x[3] = normal2param(reg_range, args.reg_lambda, float, base=10)
    x[4] = normal2param(ratio_range, args.neg_ratio, int)
    x[5] = normal2param(power_range, args.neg_power, float)
    
    return x

def plot_perf(all_scores):
    s = len(all_scores)
    plt.scatter(np.linspace(0,s-1,s), all_scores)
    plt.show()

    #best = []
    #for i in range(1, len(all_scores)):
    #    best.append(np.max(all_scores[:i]))
    #plt.plot(best)
    #plt.show()

# search through all folders
path = 'results'
folders = os.listdir(path)
folders = [f for f in folders if 'train' in f]
folders = sorted(folders, key=natural_key)

# get best performance and hps
all_scores = []
for f in folders:
    try:
        scores = np.load(os.path.join(path, f, 'metric.npy'), allow_pickle=True)
        all_scores.append(np.max(scores))
    except:
        print('skipped: ', f)

best_arg = np.argmax(all_scores)
best_score = np.max(all_scores)
print('best score: {}, best folder: {}'.format(best_score, best_arg))

plot_perf(all_scores)

sys.exit()

# don't do this for now...
args = get_args()
with open(os.path.join('results', folders[best_arg], 'info.yml'), 'r') as a:
    yml = yaml.safe_load(a)
    for key in yml.keys():
        setattr(args, key, yml[key])

gp_args = convert(args)
print(gp_args)

#############################
#torch = torch.empty((len(folders), 6))
#for i, f in enumerate(folders):
    #args = get_args()
    #with open(os.path.join('results', f, 'info.yml'), 'r') as a:
        #yml = yaml.safe_load(a)
        #for key in yml.keys():
            #setattr(args, key, yml[key])

    #x = convert(args)
    #np.expand_dims(x, axis=0)

    #y = np.load(os.path.join(path, f, 'metric.npy'), allow_pickle=True)
    #y = np.array([np.max(y)])

    #if i == 0:
        #x_train = x
        #y_train = y
    #else:
        #x_train = np.vstack((x_train, x))
        #y_train = np.concatenate((y_train, y), axis=0)

#np.save('x_train.npy', x_train)
#np.save('y_train.npy', y_train)



