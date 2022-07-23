import torch
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Measureperrel:
    def __init__(self,path):


        self.dir = path

        self.hit1  = {}
        self.hit3  = {}
        self.hit10 = {}
        self.mr = {}
        self.mrr = {}
        for i in range(0,48):
            self.hit1[i]=0.0
            self.hit3[i]=0.0
            self.hit10[i]=0.0
            self.mr[i]=0.0
            self.mrr[i]=0.0

    def update(self, rank,rel_num):
        if rank == 1:
            self.hit1[rel_num] += 1.0
        if rank <= 3:
            self.hit3[rel_num] += 1.0
        if rank <= 10:
            self.hit10[rel_num] += 1.0

        self.mr[rel_num]  += rank
        self.mrr[rel_num] += (1.0 / rank)
    
    #def normalize(self, num_facts):
    def normalize(self, rel_num,num_facts):
        if num_facts>0:
            self.hit1[rel_num]  /= (2 * num_facts)
            self.hit3[rel_num]  /= (2 * num_facts)
            self.hit10[rel_num] /= (2 * num_facts)
            self.mr[rel_num]    /= (2 * num_facts)
            self.mrr[rel_num]   /= (2 * num_facts)
        else:
            self.hit1[rel_num]=-1
            self.hit3[rel_num]=-1
            self.hit10[rel_num]=-1
            self.mr[rel_num] =-1
            self.mrr[rel_num]=-1

    def pass_back(self):
        return self.hit1, self.hit3, self.hit10, self.mr, self.mrr

    def print_(self):
        with open(self.dir +'hitone.pkl', 'wb') as f:
            pickle.dump(self.hit1,f)
        with open(self.dir +'hitthree.pkl', 'wb') as f:
            pickle.dump(self.hit3,f)
        with open(self.dir +'hitten.pkl', 'wb') as f:
            pickle.dump(self.hit10,f)
        with open(self.dir +'mr.pkl', 'wb') as f:
            pickle.dump(self.mr,f)
        with open(self.dir +'mrr.pkl', 'wb') as f:
            pickle.dump(self.mrr,f)
        
        hitones=[]
        hitthrees=[]
        hittens=[]
        mrs=[]
        mrrs=[]
        rels=[]
            
        for rel_num in range(0,48):
            rels.append(rel_num)
            print("\tNumber of Relation:",rel_num)
            print("\tHit@1 =",  self.hit1[rel_num])
            hitones.append(self.hit1[rel_num])
            print("\tHit@3 =",  self.hit3[rel_num])
            hitthrees.append(self.hit3[rel_num])
            print("\tHit@10 =", self.hit10[rel_num])
            hittens.append(self.hit10[rel_num])
            print("\tMR =",     self.mr[rel_num])
            mrs.append(self.mr[rel_num])
            print("\tMRR =",    self.mrr[rel_num])
            mrrs.append(self.mrr[rel_num])
            print("")
        #plt.figure(figsize=(24,18))
        #sns.barplot(x=rels, y=hitones)
        #plt.savefig(self.dir + 'perrel1.jpg')
        #sns.barplot(x=rels, y=hitthrees)
        #plt.savefig(self.dir  + 'perrel3.jpg')
        #sns.barplot(x=rels, y=hittens)
        #plt.savefig(self.dir + 'perrel10.jpg')
        #sns.barplot(x=rels, y=mrs)
        #plt.savefig(self.dir+ 'perrelmr.jpg')
        #sns.barplot(x=rels, y=mrrs)
        #plt.savefig(self.dir+ 'perrelmrr.jpg')

        return self.hit1, self.hit3, self.hit10, self.mr, self.mrr
            
    