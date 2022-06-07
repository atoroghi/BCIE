import os, sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

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
    
    # make plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

    # load and plot cdf for all results
    for f in load_files:
        rank = np.load(os.path.join(path, f, 'rank.npy'), allow_pickle=True)
        
        # kth element gives % of hits at k
        hit_count =  get_hit_count(rank, 50) / rank.shape[0]
        hit_cdf = np.cumsum(hit_count)
        
        ax.plot(hit_cdf)
    plt.title('CDF Hits at K')
    plt.xlabel('Rank')
    plt.ylabel('Hits')
    plt.savefig('results/cdf_hits.jpg')

    plt.show()