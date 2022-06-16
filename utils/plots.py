import os
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

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
    
    np.save(os.path.join('results', test_name, 'rec_score.npy'), np.array(rec), allow_pickle=True)
    np.save(os.path.join('results', test_name, 'kg_score.npy'), np.array(kg), allow_pickle=True)
    np.save(os.path.join('results', test_name, 'reg.npy'), np.array(reg), allow_pickle=True)

def rank_save(rank, test_name, shuffle=False):
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
    if not shuffle:
        np.save(os.path.join('results', test_name, 'rank.npy'), np.array(rank), allow_pickle=True)
    else:
        np.save(os.path.join('results', test_name, 'rank_shuffled.npy'), np.array(rank), allow_pickle=True)

