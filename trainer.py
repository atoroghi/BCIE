import os, time, sys
from models.SimplE import SimplE
from models.ComplEx import ComplEx
from utils import loss_save
from models.WRMF_torch import wrmf
from models.SVD_torch import svd
from models.POP_torch import pop
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tester import test
from dataload import DataLoader
from utils.plots import RankTrack, temporal_plot

def train(dataloader, args, device='cuda'):
    # get model, dataset and optimizer
    if args.model_type == 'wrmf':
        wrmf(dataloader, args, 'val', device)

    elif args.model_type == 'svd':
        # train model
        print('reviving svd')
        svd(dataloader, args, 'val', device)
    elif args.model_type == 'pop':
        pop(dataloader, args, 'val', device)
    elif args.model_type == 'simple' or args.model_type == 'complex':
        model = SimplE(dataloader, args, device)
        if args.model_type == 'complex':
            model = ComplEx(dataloader, args, device)
        if args.optim_type == 'adagrad':
            optimizer = torch.optim.Adagrad(
                model.parameters(), lr=args.lr, initial_accumulator_value=0.1)
        elif args.optim_type == 'adam':
            optimizer = torch.optim.Adagrad(model.parameters(), lr = args.lr)
        #path = os.path.join('results', str(args.test_name))
        path = str(args.test_name)
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'models'), exist_ok=True)

        # main training loop
        rec_score_track, kg_score_track, reg_track = [], [], []
        for epoch in range(args.epochs + 1):
            rec_score_temp, kg_score_temp, reg_temp = [], [], []

            dataloader.shuffle()
            for i in range(dataloader.n_batches):
                # ~ 3 million triplets
                #if i * dataloader.batch_size >= 500000: break

                x = dataloader.get_batch(i)
                x = x.to(device)
                optimizer.zero_grad()

                # seperate into rec and kg 
                rec_x = x[x[:,1] == 0]
                kg_x = x[x[:,1] > 0]

                # scores and loss
                rec_score = model(rec_x[:,0], rec_x[:,1], rec_x[:,2])
                kg_score = model(kg_x[:,0], kg_x[:,1], kg_x[:,2])

                rec_score_loss = model.loss(rec_score, rec_x[:,3])
                kg_score_loss = args.kg_lambda * model.loss(kg_score, kg_x[:,3])
                reg_loss = args.reg_lambda * model.reg_loss()

                loss = rec_score_loss + kg_score_loss + reg_loss

                # step
                loss.backward()
                optimizer.step()

                # save loss information
                rec_score_temp.append(rec_score_loss.cpu().item())
                kg_score_temp.append(kg_score_loss.cpu().item())
                reg_temp.append(reg_loss.cpu().item())

            rec_score_track.append(np.mean(rec_score_temp))
            kg_score_track.append(np.mean(kg_score_temp))
            reg_track.append(np.mean(reg_temp))

            #print('epoch: {}\trec score: {:.3f}\tkg score: {:.3f}\treg: {:.3f}'.format(epoch, rec_score_track[-1], kg_score_track[-1], reg_track[-1]))

            # save and test
            if epoch % args.save_each == 0:
                test(model, dataloader, epoch, args, 'val', device)

                # loss saving 
                loss_save(rec_score_track, kg_score_track, reg_track, str(args.test_name))

                # check for early stopping
                #stop_metric = np.load(os.path.join('results', str(args.test_name), 'stop_metric.npy'))
                stop_metric = np.load(os.path.join(str(args.test_name), 'stop_metric.npy'))
                best = np.argmax(stop_metric)

                if stop_metric.shape[0] - (best + 1) >= args.stop_width or epoch == args.epochs:
                    break 

                # if most recent epoch is best, save model
                if stop_metric.shape[0] - (best + 1) == 0:
                    save_path = os.path.join(path, 'models/best_model.pt')
                    print('best score: {:.7f}'.format(np.max(stop_metric)))
                    torch.save(model, save_path)

        # when finished, save metric over epoch plot
        #temporal_plot(args.test_name, k=10)