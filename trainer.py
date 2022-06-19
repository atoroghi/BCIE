import os, time, sys
from SimplE import SimplE
from utils import loss_save, perrel_save
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tester import test
from testerperreloptimized import Testerperreloptimized
from measureperrel import Measureperrel
from dataload import DataLoader

def train(dataloader, args, device='cuda'):
    # get model, dataset and optimizer
    model = SimplE(
        dataloader.num_item, dataloader.num_rel, 
        args.emb_dim, args.reg_lambda, device
    )

    optimizer = torch.optim.Adagrad(
        model.parameters(), lr = args.lr,
        weight_decay = 0, initial_accumulator_value = 0.1 
    )

    path = os.path.join('results', args.test_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'models'), exist_ok=True)

    # main training loop
    # TODO: clean up all the tracking code (it's a bit sloppy)
    rec_score_track, kg_score_track, reg_track = [], [], []
    hit1s, hit3s, hit10s, mrs, mrrs = {}, {}, {}, {}, {}
    for rel in range(0,dataloader.num_rel):
        hit1s[rel], hit3s[rel], hit10s[rel], mrs[rel], mrrs[rel]= [],[],[],[],[]

    for epoch in range(args.epochs + 1):
        rec_score_temp, kg_score_temp, reg_temp = [], [], []

        dataloader.shuffle()
        for i in tqdm(range(dataloader.n_batches)):
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

            loss = (rec_score_loss + kg_score_loss + reg_loss)

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
        print('epoch: {}\trec score: {:.3f}\tkg score: {:.3f}\treg: {:.3f}'.format(epoch, rec_score_track[-1], kg_score_track[-1], reg_track[-1]))

        # save and test
        if epoch % args.save_each == 0 and epoch != 0:
            print('testing')
            test(model.cpu(), dataloader, epoch, args, device='cpu')
            #performing test on other relations with the filtered setting
            hit1,hit3,hit10,mr,mrr = Testerperreloptimized(dataloader,model.cpu(),args,epoch)
            for rel in range(0,dataloader.num_rel):
                hit1s[rel].append(hit1[rel])
                hit3s[rel].append(hit3[rel])
                hit10s[rel].append(hit10[rel])
                mrs[rel].append(mr[rel])
                mrrs[rel].append(mrr[rel])
            model.to(device)

            print('saving model')
            save_path = os.path.join(path, 'models/epoch_{}.chkpnt'.format(epoch))
            torch.save(model, save_path)

            # loss saving 
            loss_save(rec_score_track, kg_score_track, reg_track, args.test_name)
            perrel_save(hit1s,hit3s,hit10s,mrs,mrrs,args.test_name)

            # check for early stopping
            epoch_rank = np.load(os.path.join('results', args.test_name, 'epoch_rank.npy'))
            best = np.argmax(epoch_rank)

            if epoch_rank.shape[0] - (best + 1) >= args.stop_width:
                print('early stopping')
                break 
