import os
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

def loss_save(loss, test_name):
    plt.style.use('seaborn')

    fig = plt.figure(figsize=(6,5))
    fig1 = fig.add_subplot(111)

    fig1.plot(loss, 'k')
    fig1.set_title('Loss')
    fig1.set_xlabel('Epoch')
    fig1.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', test_name, 'loss.jpg'))
    plt.clf()
    
    np.save(os.path.join('results', test_name, 'loss.npy'), np.array(loss), allow_pickle=True)

def rank_save(rank, test_name):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(6,5))
    fig1 = fig.add_subplot(111)

    c = sns.color_palette('Set2')
    sns.histplot(rank, bins=40, color=c[4])
    fig1.set_title('Pre-Critiquing Rank Distribution')
    fig1.set_xlabel('Rank')
    fig1.set_ylabel('Hits')

    plt.tight_layout()
    plt.savefig(os.path.join('results', test_name, 'rank_hist.jpg'))
    plt.clf()
    
    np.save(os.path.join('results', test_name, 'rank.npy'), np.array(rank), allow_pickle=True)

