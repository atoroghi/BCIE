import os, pickle, sys, argparse, yaml
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataload import DataLoader
from tester import get_array, get_emb, get_scores, get_rank, GetGT

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_name', default='dev', type=str, help='name of folder where model is')
    parser.add_argument('-dataset', default='ML_FB', type=str, help='wordnet dataset')
    parser.add_argument('-fold', default=0, type=int, help='fold number')
    parser.add_argument('-alpha', default=0.01, type=float, help='Learning rate for Laplace Approximation')
    parser.add_argument('-etta', default=1.0, type=float, help='Precision for Laplace Approximation')
    parser.add_argument('-mode', default='diff', type=str, help='Mode of critiquing')
    parser.add_argument('-target', default='object', type=str, help='Target of users critique')
    parser.add_argument('-num_users', default=100, type=int, help='number of users')
    parser.add_argument('-init_sigma', default=1e-2, type=float, help='initial prior precision')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parameter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model and get parameter from file
    path = os.path.join('results', args.test_name, 'fold_{}'.format(args.fold))
    with open(os.path.join(path, 'info.yml'), 'r') as f:
        yml = yaml.safe_load(f)
        for key in yml.keys():
            setattr(args, key, yml[key])

    # print important hps
    print('alpha: {}\tetta: {}\tcritique mode: {}\tcritique target: {}'.format(
        args.alpha, args.etta, args.mode, args.target))

    # load datasets
    print('loading dataset: {}'.format(args.dataset))
    dataloader = DataLoader(args)

    # load model
    print('loading model from: {}\tfold: {}'.format(args.test_name, args.fold))
    path = os.path.join('results', args.test_name, 'fold_{}'.format(args.fold))
    load_path = os.path.join(path, 'models', 'best_model.pt')
    model = torch.load(load_path).to(device)

    # load random dictionaries and such

    # make arrays with embeddings, and dict to map 
    rec_h, rec_t, id2index = get_array(model, dataloader, args, rec=True)
    item_emb = (rec_h, rec_t)

    # TODO: setup for tests??
    # load gt info and track rank scores classes
    get_gt = GetGT(dataloader.fold, 'train')

    # get all relationships
    with torch.no_grad():
        rels = dataloader.num_rel
        r = torch.linspace(0, rels - 1, rels).long().to('cuda')
        rel_f = model.rel_embs(r)
        rel_inv = model.rel_inv_embs(r)
        rel_emb = torch.stack((rel_f, rel_inv))
        rel_emb = torch.permute(rel_emb, (1,0,2))

    # all users, gt selects test / train gts
    data = dataloader.rec_train
    all_users = torch.tensor(data[:,0]).to('cuda')

    # main test loop
    for i, user in enumerate(all_users):
        # TODO: abstract this process in tester and call one function
        test_emb = get_emb(user, model)
        ranked = get_scores(test_emb, rel_emb[0], item_emb, dataloader, args.learning_rel)
        test_gt, all_gt, train_gt = get_gt.get(user.cpu().item())
        # for calculating R-precision
        
        if test_gt == None: continue
        ranks = get_rank(ranked, test_gt, all_gt, id2index[i])
        print(ranks)

    print(rec_h.shape)
    sys.exit()

    # critique loop
    # rec class and post. update class
    #for each user:
    #    get embedding 
    #    compute all scores, get top k items to rec 
    #    get test items that user likes gt

    #    get ranks of all gts 
    #    some hp / method where user make a critique (gives info)
    #    update user embedding 

    #    save rank as we update the user emb 

