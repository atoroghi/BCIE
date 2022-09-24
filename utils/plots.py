import os, sys, pickle, re
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# class to save all rank infomation to plot and save
class RankTrack:
    def __init__(self):
        self.info = {'rank':{}, 'rprec':{}}
    
    def update(self, rank, rprec, rel, k=10):
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
    path = os.path.join('results', test_name)
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

# save metric info

# save metric info
def save_metrics(rank_track, test_name, epoch, mode):
    # for train loop
    if mode == 'val':
        save_path = os.path.join('results', test_name, 'epoch_{}'.format(epoch)) 
        os.makedirs(save_path, exist_ok=True)

        # save metric_track for eval performance over training
        with open(os.path.join(save_path, 'metric_track.pkl'), 'wb') as f:
            pickle.dump(rank_track, f)

        # get hit at k for rec
        rank = rank_track.info['rank'][0] # likes relation

        rank_at_k = np.where(rank < 10)[0].shape[0] / rank.shape[0]
        stop_metric_path = os.path.join('results', test_name, 'stop_metric.npy')  

        if epoch != 0:
            scores = np.load(stop_metric_path, allow_pickle=True)
            saved_scores = np.append(scores, rank_at_k)
        else:
            saved_scores = np.array([rank_at_k])

        rprec = rank_track.info['rprec'][0] # likes relation
        avg_rprec = np.sum(rprec)/rprec.shape[0] # average r precision over all users
        print(avg_rprec)

        print(np.max(saved_scores))
        np.save(stop_metric_path, saved_scores)
        return rank_at_k

    # for test loop
    else:
        save_path = os.path.abspath(os.path.join('results', test_name, '../..'))
        os.makedirs(save_path, exist_ok=True)

        rank = rank_track.info[0] # likes relation
        hit_1 = np.where(rank < 1)[0].shape[0] / rank.shape[0]
        hit_3 = np.where(rank < 3)[0].shape[0] / rank.shape[0]
        hit_10 = np.where(rank < 10)[0].shape[0] / rank.shape[0]
        print(hit_10)

        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('{} {} {}\n'.format(hit_1, hit_3, hit_10))
            
def save_metrics_critiquing(rank_track, test_name):

    save_path = os.path.abspath(os.path.join('Critiquing_results', test_name, '../..'))
    os.makedirs(save_path, exist_ok=True)
    for session_no in range(1,6):
        rank = rank_track.info['rank'][session_no] # likes relation
        hit_1 = np.where(rank < 1)[0].shape[0] / rank.shape[0]
        hit_3 = np.where(rank < 3)[0].shape[0] / rank.shape[0]
        hit_10 = np.where(rank < 10)[0].shape[0] / rank.shape[0]
        mrr = np.sum(1 / (rank+1)) / rank.shape[0] 
        #rprec = rank_track.info['rprec'][0]
        #avg_rprec = np.sum(rprec)/rprec.shape[0]

        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            #f.write('{} {} {} {} {}\n'.format(hit_1, hit_3, hit_10, mrr, avg_rprec))
            f.write('{} {} {} {}\n'.format(hit_1, hit_3, hit_10, mrr))



# plot distribution of ranks and line plot of hits @ k per epoch
def rank_plot(rank_track, test_name, epoch):
    save_path = os.path.join('results', test_name, 'epoch_{}'.format(epoch)) 
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
    plt.savefig(os.path.join('results', test_name, 'loss.jpg'))
    plt.close()
    
    np.save(os.path.join('results', test_name, 'rec_loss.npy'), np.array(rec), allow_pickle=True)
    np.save(os.path.join('results', test_name, 'kg_loss.npy'), np.array(kg), allow_pickle=True)
    np.save(os.path.join('results', test_name, 'reg_loss.npy'), np.array(reg), allow_pickle=True)

def perrel_save(hit1s,hit3s,hit10s,mrs,mrrs,test_name):
    epoch_list=np.arange(1,1+len(hit1s[0]))
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(24,12))
    fig1 = fig.add_subplot(231)
    fig2 = fig.add_subplot(232)
    fig3 = fig.add_subplot(233)
    fig4 = fig.add_subplot(234)
    fig5 = fig.add_subplot(236)
    for rel in hit1s.keys():
        if rel!=0:
            fig1.errorbar(epoch_list, hit1s[rel], fmt='--o',color='b', label='kg rels')
            fig2.errorbar(epoch_list, hit3s[rel], fmt='--o',color='b', label='kg rels')
            fig3.errorbar(epoch_list, hit10s[rel], fmt='--o',color='b', label='kg rels')
            fig4.errorbar(epoch_list, mrs[rel], fmt='--o',color='b', label='kg rels')
            fig5.errorbar(epoch_list, mrrs[rel], fmt='--o',color='b', label='kg rels')
        else:
            fig1.errorbar(epoch_list, hit1s[rel], fmt='--o',color='r', label='likes')
            fig2.errorbar(epoch_list, hit3s[rel], fmt='--o',color='r', label='likes')
            fig3.errorbar(epoch_list, hit10s[rel], fmt='--o',color='r', label='likes')
            fig4.errorbar(epoch_list, mrs[rel], fmt='--o',color='r', label='likes')
            fig5.errorbar(epoch_list, mrrs[rel], fmt='--o',color='r', label='likes')
    fig1.set_title('Hits @ 1 for all Relations per Epoch')
    fig1.set_xlabel('Epoch')
    fig1.set_ylabel('Hits')
    _, labels1 = fig1.get_legend_handles_labels()
    labels_1=set(labels1)
    fig1.legend(labels_1)
    fig2.set_title('Hits @ 3 for all Relations per Epoch')
    fig2.set_xlabel('Epoch')
    fig2.set_ylabel('Hits')
    _, labels2 = fig2.get_legend_handles_labels()
    labels_2=set(labels2)
    fig2.legend(labels_2)
    fig3.set_title('Hits @ 10 for all Relations per Epoch')
    fig3.set_xlabel('Epoch')
    fig3.set_ylabel('Hits')
    _, labels3 = fig3.get_legend_handles_labels()
    labels_3=set(labels3)
    fig3.legend(labels_3)
    fig4.set_title('MR for all Relations per Epoch')
    fig4.set_xlabel('Epoch')
    fig4.set_ylabel('Hits')
    _, labels4 = fig4.get_legend_handles_labels()
    labels_4=set(labels4)
    fig4.legend(labels_4)
    fig5.set_title('MRR for all Relations per Epoch')
    fig5.set_xlabel('Epoch')
    fig5.set_ylabel('Hits')
    _, labels5 = fig5.get_legend_handles_labels()
    labels_5=set(labels5)
    fig5.legend(labels_5)
    plt.tight_layout()
    plt.savefig(os.path.join('results', test_name, 'Perreltests.jpg'))
    plt.clf()
