
import os, sys, re, pickle, torch
import numpy as np
from numpy.random import default_rng

import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import time
import sys, os
import pickle
















if __name__ == '__main__':
    kg_path = 'datasets/www_data/www_data/AmazonBook/kg/train.dat'
    rec_path = 'datasets/www_data/www_data/AmazonBook/rs/ratings.txt'

    kg = np.genfromtxt(kg_path, delimiter='\t', dtype=np.int32)
    rec = np.genfromtxt(rec_path, delimiter='\t', dtype=None)

    rec_users = []
    rec_items = []
    for i in range(rec.shape[0]):
        if int(rec[i][2]) >= 4:
            rec_users.append(re.search('(?<=\')(.*?)(?=\')', str(rec[i][0])).group(0))
            rec_items.append(re.sub("^0+", "", re.search('(?<=\')(.*?)(?=\')', str(rec[i][1])).group(0)))


    
    kg[:,1] += 1 # offset
    TOTAL_FB_IDS = np.max(kg) # total number of default kg pairs (# rel << # entities)

    item2kg_path = 'datasets/www_data/www_data/AmazonBook/rs/i2kg_map.tsv'
    emap_path = 'datasets/www_data/www_data/AmazonBook/rs/e_map.dat'
    ml2fb_map = {}
    with open(item2kg_path) as f:
        for line in f:
            ml_id = re.search('(.+?)\t', line)
            fb_http = re.search('\t(.+?)\n', line)
            ml2fb_map.update({re.sub("^0+", "", ml_id.group(1)) : fb_http.group(1)})

    # maps free base html links to free base id's (final format)
    id2html_map = {}
    fb2id_map = {}
    with open(emap_path) as f:
        for kg_id, line in enumerate(f):
            fb_http = re.search('\t(.+?)\n', line)
            fb2id_map.update({fb_http.group(1) : kg_id})
            id2html_map.update({kg_id : fb_http.group(1)})

    rec_users_kept = []
    rec_items_converted = []
    # convert movielens id's to freebase id's
    i = 0
    j = 0
    while True:
        if i == len(rec_items):
            break
        if rec_items[i] in ml2fb_map: 
            # get correct freebase id from data
            fb_http = ml2fb_map[rec_items[i]]
            fb_id = fb2id_map[fb_http]
            rec_items_converted.append(fb_id)
            rec_users_kept.append(rec_users[i])
            i += 1
    
        j += 1
        print("1",j)