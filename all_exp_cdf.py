import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# count hits at each k
# kth index is total hits at k
def get_hit_count(rank, max_hit):
    hit = np.zeros(max_hit)
    for i in range(max_hit):
        hit[i] = np.where(rank == i)[0].shape[0]
    return hit

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('using all test results')

        # get correct load folders
        load_files = []
        path = 'results'
        files = os.listdir(path)
        for f in files:
            # look for folders with valid result information
            if os.path.isdir(os.path.join(path, f)): 
                sub = os.listdir(os.path.join(path, f))
                if 'rank.npy' in sub:
                    load_files.append(f)
    else:
        print('this is not implimented yet...')
        sys.exit()
    load_files.sort(key=natural_keys)

    # make plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    # color maker
    #f = 0.3
    #colors = []
    #for i in range(4):
        #colors.append((0, f*i, 1))
    #for i in range(5):
        #colors.append((1, 0, f/1.2*i))

    # load and plot cdf for all results
    for i, f in enumerate(load_files):
        rank = np.load(os.path.join(path, f, 'rank.npy'), allow_pickle=True)
        
        # kth element gives % of hits at k
        hit_count =  get_hit_count(rank, 100) / rank.shape[0]
        hit_cdf = np.cumsum(hit_count)
        
        ax.plot(hit_cdf)#, color=colors[i])

    plt.title('CDF Hits at K')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Hits')
    plt.legend(load_files)
    plt.savefig('results/cdf_hits.pdf')
    plt.show()