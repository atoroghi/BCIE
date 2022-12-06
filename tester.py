import torch, os, sys, time, pickle
import numpy as np
from utils.plots import RankTrack, rank_plot, save_metrics

# make array of all items to suggest (rec or all possible head tail)
def get_array(model, dataloader, args, device, rec):
    if rec:
        all_items = torch.tensor(dataloader.rec_train[:,2]).to(device)
        items = torch.unique(all_items, sorted=False)
    else:
        print('error, kg tester not implimented')
        sys.exit()
        items = np.unique(np.concatenate((
            dataloader.rec[:,0], dataloader.rec[:,2], 
            dataloader.kg[:,0], dataloader.kg[:,2]))).tolist()
    
    # get output array
    out = make_array(model, items, args.emb_dim)
    return out

# make array with all item embeddings and dict map {item id : array index}
def make_array(model, items, emb_dim):
    # to map id to array location
    id2index = dict(zip(items.cpu().tolist(), list(range(0, len(items)))))
    index2id = {v: k for k, v in id2index.items()}

    with torch.no_grad():
        items_temp = items.long()
        items_h = model.ent_h_embs(items_temp)
        items_t = model.ent_t_embs(items_temp)

    return items_h, items_t, id2index, index2id

def get_scores(test_emb, rel_emb, item_emb, learning_rel):
    # get score, based on if test item is head or tail
    with torch.no_grad():
        if learning_rel == 'freeze':
            for_prod = torch.sum(test_emb[0] * item_emb[1], axis=1)
            scores = torch.clip(for_prod, -40, 40)
        else:
            for_prod = torch.sum(test_emb[0] * rel_emb[0] * item_emb[1], axis=1)
            inv_prod = torch.sum(test_emb[1] * rel_emb[1] * item_emb[0], axis=1)

            scores = torch.clip((for_prod + inv_prod) / 2, -40, 40)

        ranked = torch.argsort(scores, descending=True)
    return (scores, ranked)

# get ground truth
class GetGT:
    def __init__(self, fold, mode):
        path = os.path.join('datasets', 'ML_FB', 'fold {}'.format(fold))
        #names = ['kg_head_test', 'kg_head_train', 'kg_tail_test', 'kg_tail_train',
        #         'user_likes_test', 'user_likes_train']
        if mode == 'val': names = ['ul_train', 'ul_val']
        if mode == 'test': names = ['ul_train', 'ul_test', 'ul_val']
        
        self.mode = mode
        
        self.maps = []
        for n in names:
            with open(os.path.join(path, n + '.pkl'), 'rb') as f:
                self.maps.append(pickle.load(f))

    def get(self, test_item):
        if self.mode == 'val':
            train_gt = self.maps[0][test_item]
            val_gt = self.maps[1][test_item]
            return val_gt, train_gt + val_gt, train_gt

        else:
            train_gt = self.maps[0][test_item]
            test_gt = self.maps[1][test_item]

            # TODO: this is bad practice...
            # if user is in validation, filter this too... 
            try: 
                val_gt = self.maps[2][test_item]
                all_gt = train_gt + test_gt + val_gt
                train = train_gt + val_gt
            except:
                all_gt = train_gt + test_gt
                train = train_gt
            return test_gt, all_gt, train

# get final rank to show performance
def get_rank(ranked, test_gt, all_gt, id2index):
    ranked = ranked.cpu().numpy()
    # get array of all test and train items
    item_inds = []
    for item in all_gt:
        item_inds.append(id2index[item])
    item_inds = np.array(item_inds)
    
    # get rank for all items in test set for user
    rank = []
    for gt in test_gt:
        remove_inds = np.setdiff1d(item_inds, id2index[gt])
        pre_rank = np.where(ranked == id2index[gt])[0][0]       
        
        check_ranked = ranked[:pre_rank]
        hits = np.in1d(check_ranked, remove_inds) 
        sub = np.count_nonzero(hits)
        rank.append(pre_rank - sub)

    return np.array(rank)

# get final rank to show performance
def get_Rprec(ranked, test_gt, train_gt, id2index):
    R = len(test_gt)
    ranked = ranked.cpu().numpy()
    # get array of all test and train items
    item_inds = []
    for item in train_gt:
        item_inds.append(id2index[item])
    item_inds = np.array(item_inds)
    
    # get rank for all items in test set for user
    rank = []
    for gt in test_gt:
        #shouldn't this be np.setdiff1d(item_inds, id2index[gt])? isn't gt an id here?
        remove_inds = np.setdiff1d(item_inds, gt)
        pre_rank = np.where(ranked == id2index[gt])[0][0]       
        
        check_ranked = ranked[:pre_rank]
        hits = np.in1d(check_ranked, remove_inds) 
        sub = np.count_nonzero(hits)
        rank.append(pre_rank - sub)
    test_ranks = np.array(rank)
    # this is the Rprec per user
    Rprec = ((np.where((test_ranks < R)))[0].shape[0]) / R
    return Rprec

# get embedding for a head or tail id
def get_emb(test_item, model, device):
    # get embedding for test item
    with torch.no_grad():
        t = torch.tensor([test_item]).long().to(device)
        head = model.ent_h_embs(t)
        tail = model.ent_t_embs(t)
        test_emb = (head, tail)

    return test_emb

def test(model, dataloader, epoch, args, mode, device):
    assert mode in ['test', 'val']
    # get arrays with all items for link prediction
    # special array for < user, likes, ? >

    #kg_h, kg_t, kg_id2index, _ = get_array(model, dataloader, args, rec=False)
    rec_h, rec_t, rec_id2index, _ = get_array(model, dataloader, args,device, rec=True)
    kg_id2index = {}
    id2index = (rec_id2index, kg_id2index)

    # get all relationships
    with torch.no_grad():
        rels = dataloader.num_rel
        r = torch.linspace(0, rels - 1, rels).long().to('cuda')
        rel_f = model.rel_embs(r)
        rel_inv = model.rel_inv_embs(r)
        rel_emb = torch.stack((rel_f, rel_inv))
        rel_emb = torch.permute(rel_emb, (1,0,2))

    # all users to test rec on and all test triples to test kg on
    data = dataloader.rec_test if mode == 'test' else dataloader.rec_val

    all_users = torch.tensor(data[:,0]).to('cuda')
    users = torch.unique(all_users, sorted=False)
    kg_triples = None #dataloader.kg_test 

    # load gt info and track rank scores classes
    get_gt = GetGT(dataloader.fold, mode)
    rank_track = RankTrack()

    # NOTICE: KG IS NOT BEING USED 
    # main test loop
    for i, test_items in enumerate((users, kg_triples)):
        if i == 1: break
        #if i == 1 and args.kg == 'no_kg': break
        item_emb = (rec_h, rec_t) #if i == 0 else (kg_h, kg_t)

        for j, test_item in enumerate(test_items):
            if j > 100: break
            #if j%100 == 0: print('{:.5f} {:.5f}'.format(j/test_items.shape[0], (time.time()-t0) / 60))

            # for rec testing
            if i == 0:
                test_emb = get_emb(test_item, model, device)
                _, ranked = get_scores(test_emb, rel_emb[0], item_emb, args.learning_rel)
                test_gt, all_gt, train_gt = get_gt.get(test_item.cpu().item())

                # for calculating r-precision
                if test_gt == None: continue
                ranks = get_rank(ranked, test_gt, all_gt, id2index[i])
                rprec = get_Rprec(ranked, test_gt, train_gt, id2index[i])

                rank_track.update(ranks, rprec, 0)
            
            else:
                print('testing here not implimented...')
                rel = test_item[1]
                
                # test_item as head
                test_emb = get_emb(test_item[0], model, device)
                ranked = get_scores(test_emb, rel_emb[rel], item_emb, dataloader)
                test_gt, all_gt = get_gt.get(test_item[0], rel, head=True)
                ranks = get_rank(ranked, test_gt, all_gt, id2index[i])
                rank_track.update(ranks, rel)

                # test_item as tail
                test_emb = get_emb(test_item[2], model, device)
                ranked = get_scores(test_emb, rel_emb[rel], item_emb, dataloader)
                ranks = get_rank(ranked, test_gt, all_gt, id2index[i])
                rank_track.update(ranks, rel)

    # different save options if train or testing        
    if mode == 'val': 
        rank_plot(rank_track, args.test_name, epoch)
        save_metrics(rank_track, args.test_name, epoch, mode)
    else:
        save_metrics(rank_track, args.test_name, epoch, mode)