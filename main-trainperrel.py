from testerperreloptimized import Testerperreloptimized
from dataset import Dataset
from trainer import train
from measureperrel import Measureperrel
from recommender import Recommender
from dataloadperrel import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import torch
import pickle
import seaborn as sns
import matplotlib.pyplot as plt 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help="folder for test results")
    parser.add_argument('-batch_size', default=16384, type=int, help="batch size")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-neg_power', default=0.0, type=float, help="power for neg sampling disribution")
    parser.add_argument('-sample_type', default='single', type=str, help="single or double (double treats head and tail dists differently)")

    parser.add_argument('-epochs', default=5, type=int, help="number of epochs")
    parser.add_argument('-save_each', default=None, type=int, help="validate every k epochs")
    parser.add_argument('-workers', default=1, type=int, help="threads for dataloader")
    parser.add_argument('-emb_dim', default=64, type=int, help="embedding dimension")

    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.01, type=float, help="l2 regularization parameter")

    # TODO: fix this...
    parser.add_argument('-ni', default=0, type=float, help="noise intensity")
    parser.add_argument('-dataset', default="ML_FB", type=str, help="wordnet dataset")
    parser.add_argument('-max_iters_laplace', default=1000, type=int, help="Maximum number of iterations for Laplace Approximation")
    parser.add_argument('-alpha', default=0.01, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-etta', default=1.0, type=float, help="Learning rate for Laplace Approximation")
    parser.add_argument('-kg_inclusion', default=True, type=bool, help="Whether we want the kg data to be included or not")
    args = parser.parse_args()
    return args
def save_hyperparams(path, args):
    with open(os.path.join(path, 'info.txt'), 'w') as f:
        f.write('batch size: {}\n'.format(args.batch_size))
        f.write('epochs: {}\n'.format(args.epochs))
        f.write('learning rate: {}\n'.format(args.lr))
        f.write('lambda regularizer: {}\n'.format(args.reg_lambda))
        f.write('dataset: {}\n'.format(args.dataset))
        f.write('embedding dimension: {}\n'.format(args.emb_dim))
        f.write('negative ratio: {}\n'.format(args.neg_ratio))
        f.write('negative power: {}\n'.format(args.neg_power))
        f.write('alpha: {}\n'.format(args.alpha))
        f.write('etta: {}\n'.format(args.etta))
        f.write('noise intensity: {}\n'.format(args.ni))
        f.write('max laplace iterations: {}\n'.format(args.max_iters_laplace))


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(args)
    #plotter= Plotter(history,session_length)
    #plotter.plot_ranks()

    #print("~~~~ Training ~~~~")
    #trainer = Trainer(dataset, args)
    #trainer.train()
    #train(dataloader, args, device)

    #print("~~~~ Select best epoch on validation set ~~~~")
    #epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
   # dataset = Dataset(args.dataset,args.ni)
    
    #best_mrr = -1.0
    #best_epoch = "0"
    #for epoch in epochs2test:
      #  start = time.time()
     #   print(epoch)
        #model_path = "models/" + args.dataset + "/optimizerswithKG/" +self.args.num_run+"/" +epoch + ".chkpnt"
        #tester = Tester(dataset, model_path, "valid")
        #mrr = tester.test()
        #if mrr > best_mrr:
        #    best_mrr = mrr
        #    best_epoch = epoch
       # print(time.time() - start)

    #print("Best epoch: " + best_epoch)

    hitones=[]
    hitthrees=[]
    hittens=[]
    mrs=[]
    mrrs=[]

    epochs=[5,20]

    print("~~~~ Testing on different relations ~~~~")
    for epoch in epochs:
        #best_model_path ="results/perrels/" + args.test_name + "/models/" +str(args.epochs) + ".chkpnt"
        best_model_path ="results/perrels/" + args.test_name + str(epoch)+ "/models/" + str(epoch) + ".chkpnt"
        tester = Testerperreloptimized(dataloader, best_model_path, "test",args,epoch)
        hit1,hit3,hit10,mr,mrr = tester.test()
        hitones.append(hit1)
        hitthrees.append(hit3)
        hittens.append(hit10)
        mrs.append(mr)
        mrrs.append(mrr)
    def plot_save(x, y1, name):
        plt.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        ax.errorbar(x, y1, fmt='-o')
        plt.rcParams.update({'font.size': 20})
        ax.set_xlabel('Epochs', fontsize=20)
        ax.set_ylabel(name, fontsize=20)
        plt.tight_layout()
        plt.savefig('results/perrels'+name+ '.jpg')
        plt.clf()
    

    plot_save(epochs,mrs,'MR')
    plot_save(epochs,mrrs,'MRR')
    plot_save(epochs,hitones,'hit@1')
    plot_save(epochs,hitthrees,'hit@3')
    plot_save(epochs,hittens,'hit@10')
