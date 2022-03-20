import torch

class Measure:
    def __init__(self, user_embedding=[], likes_embedding=[], items=[], ground_truth=0, emb_dim=20):


        self.user_embedding = user_embedding
        self.likes_embedding = likes_embedding
        self.items = items
        self.ground_truth = ground_truth
        self.emb_dim = emb_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        self.hit1  = {"raw": 0.0, "fil": 0.0}
        self.hit3  = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr   = {"raw": 0.0, "fil": 0.0}
        self.mr    = {"raw": 0.0, "fil": 0.0}


    def get_rank(self):

        candidates = torch.zeros(len(self.items),self.emb_dim)
        candidate_counter = 0
        ground_truth_indicator= 0

        for item in self.items:
            if item == self.ground_truth:
                # to know which row is the ground truth stored in
                ground_truth_indicator= candidate_counter
            t= torch.tensor([item]).long().to(self.device)
            k= self.likes_embedding*t
            candidates[candidate_counter]= k
            candidate_counter+= 1
        print("candidates tensor:"+str(candidates))

        logits_all= torch.mv(candidates,self.user_embedding)
        print("logits_all:"+str(logits_all))
        logits_gt= logits_all[ground_truth_indicator]
        print("logits_gt:"+str(logits_gt))
        gt_rank= sum([logit>=logits_gt for logit in logits_all])
        return gt_rank



    def update(self, rank, raw_or_fil):
        if rank == 1:
            self.hit1[raw_or_fil] += 1.0
        if rank <= 3:
            self.hit3[raw_or_fil] += 1.0
        if rank <= 10:
            self.hit10[raw_or_fil] += 1.0

        self.mr[raw_or_fil]  += rank
        self.mrr[raw_or_fil] += (1.0 / rank)
    
    def normalize(self, num_facts):
        for raw_or_fil in ["raw", "fil"]:
            self.hit1[raw_or_fil]  /= (2 * num_facts)
            self.hit3[raw_or_fil]  /= (2 * num_facts)
            self.hit10[raw_or_fil] /= (2 * num_facts)
            self.mr[raw_or_fil]    /= (2 * num_facts)
            self.mrr[raw_or_fil]   /= (2 * num_facts)

    def print_(self):
        for raw_or_fil in ["raw", "fil"]:
            print(raw_or_fil.title() + " setting:")
            print("\tHit@1 =",  self.hit1[raw_or_fil])
            print("\tHit@3 =",  self.hit3[raw_or_fil])
            print("\tHit@10 =", self.hit10[raw_or_fil])
            print("\tMR =",     self.mr[raw_or_fil])
            print("\tMRR =",    self.mrr[raw_or_fil])
            print("")
    