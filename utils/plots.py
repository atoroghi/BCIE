import os, sys
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

# class to save all rank infomation to plot and save
class RankTrack:
    def __init__(self):
        self.info = {}
    
    def update(self, rank, rel):
        if rel not in self.info:
            self.info.update({rel : rank})
        else:
            self.info[rel] = np.concatenate((self.info[rel], rank))

    def items(self):
        return self.info.items()

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

# plot distribution of ranks and line plot of hits @ k per epoch
def rank_save(rank_track, test_name, epoch, k=100):
    save_path = os.path.join('results', test_name, 'epoch {}'.format(epoch)) 
    os.makedirs(save_path, exist_ok=True)

    # distribution
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    color = sns.color_palette('Set2')
    for k, v in rank_track.items():
        if k == 0:
            #rank_at_1 = np.where(v < 1)[0].shape[0]
            #rank_at_3 = np.where(v < 3)[0].shape[0]
            #print('rank at 1: {}'.format(rank_at_1 / v.shape[0]))
            #print('rank at 3: {}'.format(rank_at_3 / v.shape[0]))
            rank_at_10 = np.where(v < 10)[0].shape[0] / v.shape[0]
            #print('rank at 10: {}'.format(rank_at_10))
            metric_path = os.path.join('results', test_name, 'metric.npy')  
            if epoch != 0:
                scores = np.load(metric_path, allow_pickle=True)
                print(np.max(scores))
                np.save(metric_path, np.append(scores, rank_at_10))
            else:
                np.save(metric_path, np.array([rank_at_10]))

            sns.histplot(v, bins=40, ax=ax1)
        else:
            a = k % 7
            sns.histplot(v, bins=40, ax=ax2, color=color[a])

    ax1.set_title('Rec Rank Distribution')
    ax2.set_title('KG Rank Distribution')
    ax1.set_xlabel('Rank')
    ax2.set_xlabel('Rank')
    ax1.set_ylabel('Hits')
    ax2.set_ylabel('Hits')

    plt.tight_layout() 
    plt.savefig(os.path.join(save_path, 'rank_hist.jpg'))
    plt.close()

# TODO: are either of these used?
#def rank_plot():
    ## line plot
    #files = os.listdir(os.path.join('results', test_name))
    #if 'epoch_rank.npy' in files:
        #epoch_rank = np.load(os.path.join('results', test_name, 'epoch_rank.npy'))
        #rank_at_n = np.where(rank < 100)[0].shape[0]
        #epoch_rank = np.append(epoch_rank, rank_at_n)

        ## plotting
        #fig = plt.figure(figsize=(6,5))
        #fig1 = fig.add_subplot(111)

        ## x axis must be aligned to the epoch save number
        #fig1.plot(epoch * np.linspace(0, 1, epoch_rank.shape[0]), epoch_rank)

        #fig1.set_title('Hits @ 100 per Epoch')
        #fig1.set_xlabel('Epoch')
        #fig1.set_ylabel('Hits')

        #plt.tight_layout() 
        #plt.savefig(os.path.join('results', test_name, 'hit_plot.jpg'))
        #np.save(os.path.join(save_path, 'rank.npy'), np.array(rank), allow_pickle=True)
        #plt.close()

        #np.save(os.path.join('results', test_name, 'epoch_rank.npy'), epoch_rank)
    #else: 
        #epoch_rank = np.array([np.where(rank < 100)[0].shape[0]])
        #np.save(os.path.join('results', test_name, 'epoch_rank.npy'), epoch_rank)

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