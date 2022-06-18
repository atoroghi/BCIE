from genericpath import exists
import os
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

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
    plt.clf()
    
    np.save(os.path.join('results', test_name, 'rec_loss.npy'), np.array(rec), allow_pickle=True)
    np.save(os.path.join('results', test_name, 'kg_loss.npy'), np.array(kg), allow_pickle=True)
    np.save(os.path.join('results', test_name, 'reg_loss.npy'), np.array(reg), allow_pickle=True)

# plot distribution of ranks and line plot of hits @ k per epoch
def rank_save(rank, test_name, epoch):
    rank = np.array(rank)
    
    save_path = os.path.join('results', test_name, 'epoch {}'.format(epoch)) 
    os.makedirs(save_path, exist_ok=True)
    
    # distribution
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(6,5))
    fig1 = fig.add_subplot(111)

    c = sns.color_palette('Set2')
    sns.histplot(rank, bins=40, color=c[4])
    fig1.set_title('Pre-Critiquing Rank Distribution')
    fig1.set_xlabel('Rank')
    fig1.set_ylabel('Hits')

    plt.tight_layout() 
    plt.savefig(os.path.join(save_path, 'rank_hist.jpg'))
    np.save(os.path.join(save_path, 'rank.npy'), np.array(rank), allow_pickle=True)
    plt.close()

    # line plot
    files = os.listdir(os.path.join('results', test_name))
    if 'epoch_rank.npy' in files:
        epoch_rank = np.load(os.path.join('results', test_name, 'epoch_rank.npy'))
        rank_at_n = np.where(rank < 100)[0].shape[0]
        epoch_rank = np.append(epoch_rank, rank_at_n)

        # plotting
        fig = plt.figure(figsize=(6,5))
        fig1 = fig.add_subplot(111)

        # x axis must be aligned to the epoch save number
        fig1.plot(epoch * np.linspace(0, 1, epoch_rank.shape[0]), epoch_rank)

        fig1.set_title('Hits @ 100 per Epoch')
        fig1.set_xlabel('Epoch')
        fig1.set_ylabel('Hits')

        plt.tight_layout() 
        plt.savefig(os.path.join('results', test_name, 'hit_plot.jpg'))
        np.save(os.path.join(save_path, 'rank.npy'), np.array(rank), allow_pickle=True)
        plt.close()

        np.save(os.path.join('results', test_name, 'epoch_rank.npy'), epoch_rank)
    else: 
        epoch_rank = np.array([np.where(rank < 100)[0].shape[0]])
        np.save(os.path.join('results', test_name, 'epoch_rank.npy'), epoch_rank)
