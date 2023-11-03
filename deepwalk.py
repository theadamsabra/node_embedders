import torch
import torch_geometric
from torch_geometric.datasets import Flickr

class DeepWalk:
    def __init__(self, G, win_size, embedding_size, walks_per_vertex, walk_length, seed=36) -> None:
        '''
        Core DeepWalk class. Contains main algorithm from paper including SkipGram.
        Inputs:
            - G (): 
            - win_size (int): window size
            - embedding_size (int): embedding size of output representation
            - walks_per_vertex (int): number of walks per vertex
            - walk_length (int): random walk length
            - seed (int): random seed. default set to 36.
        '''
        self.G = G
        self.win_size = win_size
        self.embedding_size = embedding_size
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        self.seed = seed

    def _init_vertex_representations(self):
        torch_geometric.seed_everything(self.seed) 
        self.phi = torch.rand(
            size = (len(self.G), self.embedding_size)
        )

    def calculate_embeddings(self):
        '''Main function run to calculate embedding matrix.'''
        self._init_vertex_representations()
        pass 
