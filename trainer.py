import os
from SimplE import SimplE
from dataload import LoadDataset
#from utils import loss_plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
class Trainer:
    def __init__(self, dataset, args, device):
        self.device = device
        self.model = SimplE(dataset.num_items, 
                            dataset.num_rel, 
                            args.emb_dim,
                            args.reg_lambda,
                            self.device)
        self.dataset = dataset
        self.args = args
        
    def train(self):
        self.model.train()

        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr = self.args.lr,
            weight_decay = 0,
            initial_accumulator_value = 0.1 # this is added because of the consistency to the original tensorflow code
        )
        dataloader = DataLoader(
            self.dataset, 
            batch_size = self.args.workers, # each workers get batch_size / workers items 
            shuffle = True, 
            num_workers = self.args.workers
        )

        # main training loop
        loss_track = []
        for epoch in tqdm(range(self.args.ne+1)):
            
            loss_temp = []
            for x in dataloader:
                x = x.to(self.device)
                x = x.view(-1, 4)
                optimizer.zero_grad()

                # TODO: too many return values... this should be a class
                score,_,_,_,_,_,_ = self.model(x[:,0],x[:,1],x[:,2])
                score_loss, reg_loss = self.model.loss(score, x)
                loss = (score_loss + reg_loss)

                #scores,_,_,_,_,_,_ = self.model(h, r, t)
                #loss = torch.sum(F.softplus(-l * scores))\ 
                # + (self.args.reg_lambda * self.model.l2_loss()\
                #       / self.dataset.num_batch(self.args.batch_size))
                
                loss.backward()
                optimizer.step()
                loss_temp.append(loss.cpu().item())
            loss_track.append(np.mean(loss_temp))
            print('epoch: {}\tloss: {}'.format(epoch, loss_track[-1]))

            if epoch % self.args.save_each == 0:
                self.save_model(epoch)
        print(loss_track)
        #loss_plt(loss_track, self.args.test_name)

    def save_model(self, chkpnt):
        #print("Saving the model")
        directory = "models/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model, directory + str(chkpnt) + ".chkpnt")

