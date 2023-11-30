import torch
import multiprocessing as mp
from gensim.models import Word2Vec
from random_walk import RandomWalk
from torch_geometric.datasets import Flickr

class DeepWalk:
    def __init__(self, G, win_size, embedding_size, walks_per_vertex, walk_length, num_workers, seed=36) -> None:
        '''
        Core DeepWalk class. 
        Inputs:
            - G (torch_geometric.data.Data): torch geometric data object.   
            - win_size (int): window size
            - embedding_size (int): embedding size of output representation
            - walks_per_vertex (int): number of walks per vertex
            - walk_length (int): random walk length
            - num_workers (int): number of CPU workers for SkipGram.
            - seed (int): random seed. default set to 36.
        '''
        self.G = G
        self.num_vertices = self.G.x.shape[0]
        self.edges = self.G.edge_index
        self.win_size = win_size
        self.embedding_size = embedding_size
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        self.seed = seed
        self.random_walk = RandomWalk(self.walks_per_vertex, self.walk_length)
        self.num_workers = num_workers
        self.skipgram = Word2Vec(
            vector_size=self.embedding_size,
            window=self.win_size,
            workers=self.num_workers,
            sg=1, # Default sg to 1 to leverage SkipGram and not CBOW
            hs=1 # Default hs to 1 to leverage Heirarchical Softmax
        )

    def _shuffle(self):
        '''Shuffle vertices'''
        return torch.randperm(self.num_vertices) 

    def random_walk_core(self, O:torch.Tensor, walks) -> list:
        '''
        Construct walks for SkipGram algorithm.
        
        Inputs:
            - O (torch.Tensor): shuffled vertices of graph.
            - walks (multiprocessing Manager List): multiprocessing manager list for multiprocessing based
                updating. 
        Returns:
            - walks (list [list]): list of all random walks constructed.
        '''
        for vertex in O:
            # Get vertex in question and do random walk:
            vertex = vertex.item() 
            walk = self.random_walk(self.edges, vertex)
            walks.append(walk)

    def construct_all_walks(self):
        '''Construct all random walks and save as method.'''
        # Main loop in question:
        walks = []
        # Number of walks per vertex
        for _ in range(0, self.walks_per_vertex):
            # Shuffle V
            O = self._shuffle()
            self.random_walk_core(O, walks) 
        self.walks = walks


# Used for debugging for now:
if __name__ == "__main__":
    G = Flickr('data/flickr')
    win_size = 10
    embedding_size = 128 
    walks_per_vertex = 80
    walk_len = 40
    num_workers = 16
    deepwalk = DeepWalk(G, win_size, embedding_size, walks_per_vertex, walk_len, num_workers)
    deepwalk.construct_all_walks()