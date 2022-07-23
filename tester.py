import torch, os, sys, time, pickle
import numpy as np
from utils.plots import rank_save, RankTrack

# make array of all items to suggest (rec or all possible head tail)
def get_array(model, dataloader, args, rec):
    if rec:
        all_items = torch.tensor(dataloader.rec[:,2]).to('cuda')
        items = torch.unique(all_items, sorted=False)
    else:
        print('error')
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
    item_id2index = dict(zip(items.cpu().tolist(), list(range(0, len(items)))))

    with torch.no_grad():
        items_temp = items.long()
        items_h = model.ent_h_embs(items_temp)
        items_t = model.ent_t_embs(items_temp)

    return items_h, items_t, item_id2index

# get scores
def get_scores(test_emb, rel_emb, item_emb, dataloader):
    # get score, based on if test item is head or tail
    # embds are [head_emb, tail_emb] always
    for_prod = torch.sum(test_emb[0] * rel_emb[0] * item_emb[1], axis=1)
    inv_prod = torch.sum(test_emb[1] * rel_emb[1] * item_emb[0], axis=1)
    scores = torch.clip((for_prod + inv_prod) / 2, -40, 40)
    ranked = torch.argsort(scores, descending=True)
    return ranked

# get gt
class GetGT:
    def __init__(self):
        path = 'datasets/ML_FB'
        names = ['kg_head_test', 'kg_head_train', 'kg_tail_test', 'kg_tail_train',
                 'user_likes_test', 'user_likes_train']
        #names = ['user_likes_test' , 'user_likes_train']
        self.maps = []
        for n in names:
            with open(os.path.join(path, n + '.pkl'), 'rb') as f:
                self.maps.append(pickle.load(f))

    def get(self, test_item, rel, head=True, rec=False):
        if rec:
            test_gt = self.maps[4][test_item]
            try:
                train_gt = self.maps[5][test_item]
            except:
                #print('skip: ', test_item)
                return None, None
        else:
            if head:
                key = (test_item, rel)
                test_gt = self.maps[0][key]
                train_gt = self.maps[1][key]
            else:
                key = (rel, test_item)
                test_gt = self.maps[2][key]
                train_gt = self.maps[3][key]
        return test_gt, test_gt + train_gt

# TODO: make these more efficient...
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
        remove_inds = np.setdiff1d(item_inds, gt)
        pre_rank = np.where(ranked == id2index[gt])[0][0]       
        
        check_ranked = ranked[:pre_rank]
        hits = np.in1d(check_ranked, remove_inds) 
        sub = np.count_nonzero(hits)
        rank.append(pre_rank - sub)

    return np.array(rank)

# get embedding for a head or tail id
def get_emb(test_item, model):
    # get embedding for test item
    with torch.no_grad():
        t = torch.tensor([test_item]).long().to('cuda')
        head = model.ent_h_embs(t)
        tail = model.ent_t_embs(t)
        test_emb = (head, tail)

    return test_emb

def test(model, dataloader, epoch, args, device):
    model.to('cuda')
    # get arrays with all items for link prediction
    # special array for < user, likes, ? >
    rec_h, rec_t, rec_id2index = get_array(model, dataloader, args, rec=True)
    kg_h, kg_t, kg_id2index = get_array(model, dataloader, args, rec=False)
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
    all_users = torch.tensor(dataloader.rec_test[:,0]).to('cuda')
    users = torch.unique(all_users, sorted=False)
    kg_triples = 'idk' #dataloader.kg_test 

    # load gt info and track rank scores classes
    get_gt = GetGT()
    rank_track = RankTrack()

    t0 = time.time()

    ranks_all = []
    # main test loop
    for i, test_items in enumerate((users, kg_triples)):
        if i == 1 and args.kg == 'no_kg': break
        if i == 1: break
        #item_emb = (rec_h, rec_t) if i == 0 else (kg_h, kg_t)
        item_emb = (rec_h, rec_t)

        for j, test_item in enumerate(test_items):
            #if j%100 == 0: print('{:.5f} {:.5f}'.format(j/test_items.shape[0], (time.time()-t0) / 60))
            if i == 0 and j >= 500: break
            if i == 1 and j >= 10: break 

            # for rec testing
            if i == 0:
                test_emb = get_emb(test_item, model)
                ranked = get_scores(test_emb, rel_emb[0], item_emb, dataloader)
                test_gt, all_gt = get_gt.get(test_item.cpu().item(), dataloader, rec=True)
                if test_gt == None: continue
                ranks = get_rank(ranked, test_gt, all_gt, id2index[i])
                rank_track.update(ranks, 0)
                ranks_all.append(ranks)
            
            else:
                rel = test_item[1]
                
                # test_item as head
                test_emb = get_emb(test_item[0], model)
                ranked = get_scores(test_emb, rel_emb[rel], item_emb, dataloader)
                test_gt, all_gt = get_gt.get(test_item[0], rel, head=True)
                ranks = get_rank(ranked, test_gt, all_gt, id2index[i])
                rank_track.update(ranks, rel)

                # test_item as tail
                test_emb = get_emb(test_item[2], model)
                ranked = get_scores(test_emb, rel_emb[rel], item_emb, dataloader)
                ranks = get_rank(ranked, test_gt, all_gt, id2index[i])
                rank_track.update(ranks, rel)
        
    rank_save(rank_track, args.test_name, epoch)
    return ((ranks_all<11).sum())/len(ranks)