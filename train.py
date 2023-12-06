import os
from deepwalk import DeepWalk
from torch_geometric.datasets import Flickr
from tqdm import tqdm
import numpy as np

G = Flickr(root='data/flickr') # G can be any dataset from torch_geometric.datasets
win_size = 10
embedding_size = 128 
walks_per_vertex = 80
walk_len = 40
num_workers = 16

for percent in tqdm([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]):
    deepwalk = DeepWalk(G, win_size, embedding_size, walks_per_vertex, walk_len, num_workers, percent)
    deepwalk.calculate_embeddings()
    np.save(
        file=os.path.join('flickr_arrays', f'weights_{int(percent*100)}_percent'),
        arr=deepwalk.trained_weights
    ) 
    np.save(
        file=os.path.join('flickr_arrays', f'vertices_{int(percent*100)}_percent'),
        arr=deepwalk.trained_vertices
    )
