import os, sys, pickle, re, torch
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# class to save all rank infomation to plot and save
class RankTrack:
    def __init__(self):
        self.info = {'rank':{}, 'rprec':{}}
    
    def update(self, rank, rprec, rel):
        if rel not in self.info['rank']:
            self.info['rank'].update({rel : rank})
        else:
            self.info['rank'][rel] = np.concatenate((self.info['rank'][rel], rank))
        if rel not in self.info['rprec']:
            self.info['rprec'].update({rel : np.array([rprec])})
        else:
            self.info['rprec'][rel] = np.concatenate((self.info['rprec'][rel], np.array([rprec])))

    def items(self):
        return self.info['rank'].items()

# plots metics over the training sequence
def temporal_plot(test_name, k):
    # get all folders
    #path = os.path.join('results', test_name)
    path = str(test_name)
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folders = sorted(folders, key=natural_key)

    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # loop through each epoch
    hit_map, mrrs, labels = {}, {} , []
    for epoch in folders:
        if 'epoch' not in epoch: continue # don't look in models folder
        labels.append(int(epoch.split('_')[1]))

        # get rank_track object
        with open(os.path.join(path, epoch, 'metric_track.pkl'), 'rb') as f_:
            rank_track = pickle.load(f_)

        # calculate metrics and plot 
        for rel, rank in rank_track.items():
            if rel not in hit_map:
                hit_map.update({rel : []})
            if rel not in mrrs:
                mrrs.update({rel : []})

            rank_at_k = np.where(rank < k)[0].shape[0] / rank.shape[0]  
            mrr = np.sum(1 / rank) / rank.shape[0]  
            hit_map[rel].append(rank_at_k)
            mrrs[rel].append(mrr)

    for rel, hit in hit_map.items():
        if rel == 0: # likes relationship
            ax1.plot(labels, hit)
        else:
            ax2.plot(labels, hit)

    ax1.set_title('Rec Hits@{}'.format(k))
    ax2.set_title('KG Hits@{}'.format(k))
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Hits %')
    ax2.set_ylabel('Hits %')

    plt.tight_layout() 
    plt.savefig(os.path.join(path, 'hits_epoch.jpg'))
    plt.close()

# TODO: this is a bad function
# save metric info
def save_metrics(rank_track, test_name, epoch, mode, k=10):
    # for train loop
    if mode == 'val':
        #save_path = os.path.join('results', test_name, 'epoch_{}'.format(epoch)) 
        save_path = os.path.join(test_name, 'epoch_{}'.format(epoch)) 
        
        os.makedirs(save_path, exist_ok=True)

        # save metric_track for eval performance over training
        with open(os.path.join(save_path, 'metric_track.pkl'), 'wb') as f:
            pickle.dump(rank_track, f)

        # get hit at k for rec
        rank = rank_track.info['rank'][0] # likes relation
        mrr = np.mean(1 / (rank + 1))
        #rank_at_k = np.where(rank < k)[0].shape[0] / rank.shape[0]

        # load previous rank_tracking
        #stop_metric_path = os.path.join('results', test_name, 'stop_metric.npy')  
        stop_metric_path = os.path.join(test_name, 'stop_metric.npy')  
        
        if epoch != 0:
            scores = np.load(stop_metric_path, allow_pickle=True)
            saved_scores = np.append(scores, mrr)
        else:
            saved_scores = np.array([mrr])
        np.save(stop_metric_path, saved_scores)
        #rprec = rank_track.info['rprec'][0] # likes relation
        #avg_rprec = np.sum(rprec) / rprec.shape[0] # average r precision over all users

    # for test loop
    else:
        #save_path = os.path.abspath(os.path.join('results', test_name, '../..'))
        save_path = os.path.abspath(os.path.join(test_name, '../..'))
        
        os.makedirs(save_path, exist_ok=True)

        rank = rank_track.info[0] # likes relation
        hit_1 = np.where(rank < 2)[0].shape[0] / rank.shape[0]
        hit_3 = np.where(rank < 4)[0].shape[0] / rank.shape[0]
        hit_10 = np.where(rank < 11)[0].shape[0] / rank.shape[0]
        print(hit_10)

        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('{} {} {}\n'.format(hit_1, hit_3, hit_10))
        np.save(os.path.join(save_path, 'rank_track.npy'), rank)            

# plot distribution of ranks and line plot of hits @ k per epoch
def rank_plot(rank_track, test_name, epoch):
    #save_path = os.path.join('results', test_name, 'epoch_{}'.format(epoch)) 
    save_path = os.path.join(test_name, 'epoch_{}'.format(epoch)) 
    
    os.makedirs(save_path, exist_ok=True)

    # distribution
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    color = sns.color_palette('Set2')
    for rel, rank in rank_track.items():
        if rel == 0:
            sns.histplot(rank, bins=40, ax=ax1)
        else:
            # k was undefined here
            k = 7
            a = k % 7
            sns.histplot(rank, bins=40, ax=ax2, color=color[a])

    ax1.set_title('Rec Rank Distribution')
    ax2.set_title('KG Rank Distribution')
    ax1.set_xlabel('Rank')
    ax2.set_xlabel('Rank')
    ax1.set_ylabel('Hits')
    ax2.set_ylabel('Hits')

    plt.tight_layout() 
    plt.savefig(os.path.join(save_path, 'rank_hist.jpg'))
    plt.close()

# plot loss
def loss_save(rec, kg, reg, test_name):
    plt.style.use('seaborn')

    fig = plt.figure(figsize=(15,5))
    fig1 = fig.add_subplot(131)
    fig2 = fig.add_subplot(132)
    fig3 = fig.add_subplot(133)

    fig1.plot(rec, 'k')
    fig2.plot(kg, 'b')
    fig3.plot(reg, 'r')

    fig1.set_title('Rec Loss')
    fig2.set_title('KG Loss')
    fig3.set_title('Reg Loss')
    
    fig1.set_ylabel('Loss')
    fig2.set_xlabel('Epoch')
    
    plt.tight_layout()
    #plt.savefig(os.path.join('results', test_name, 'loss.jpg'))
    plt.savefig(os.path.join(test_name, 'loss.jpg'))
    
    plt.close()
    #np.save(os.path.join('results', test_name, 'rec_loss.npy'), np.array(rec), allow_pickle=True)
    #np.save(os.path.join('results', test_name, 'kg_loss.npy'), np.array(kg), allow_pickle=True)
    #np.save(os.path.join('results', test_name, 'reg_loss.npy'), np.array(reg), allow_pickle=True)

    np.save(os.path.join(test_name, 'rec_loss.npy'), np.array(rec), allow_pickle=True)
    np.save(os.path.join(test_name, 'kg_loss.npy'), np.array(kg), allow_pickle=True)
    np.save(os.path.join(test_name, 'reg_loss.npy'), np.array(reg), allow_pickle=True)
