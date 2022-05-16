import os, time
from SimplE import SimplE
from dataload import LoadDataset
from utils import loss_save
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train(dataset, args, device='cuda'):
    # get model, dataset and optimizer
    model = SimplE(dataset.num_items, 
                   dataset.num_rel, 
                   args.emb_dim,
                   args.reg_lambda,
                   device
    )

    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr = args.lr,
        weight_decay = 0,
        initial_accumulator_value = 0.1 
    )

    dataloader = DataLoader(
        dataset, 
        batch_size = args.workers, # Don't touch this ever! 
        shuffle = True, 
        num_workers = args.workers
    )

    path = os.path.join('results', args.test_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'models'), exist_ok=True)

    # main training loop
    loss_track = []
    for epoch in range(args.ne + 1):
        
        loss_temp = []
        for x in dataloader:
            x = x.to(device)
            x = x.view(-1, 4)
            optimizer.zero_grad()

            score = model(x[:,0], x[:,1], x[:,2])
            score_loss, reg_loss = model.loss(score, x)
            loss = (score_loss + reg_loss)

            loss.backward()
            optimizer.step()
            loss_temp.append(loss.cpu().item())

        loss_track.append(np.mean(loss_temp))
        print('epoch: {}\tloss: {}'.format(epoch, loss_track[-1]))

        if epoch % args.save_each == 0 and epoch != 0:
            print('saving model')
            save_path = os.path.join(path, 'models/epoch {}.chkpnt'.format(epoch))
            torch.save(model, save_path)
    loss_save(loss_track, args.test_name)
