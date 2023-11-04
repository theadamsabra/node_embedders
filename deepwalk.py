import torch
import torch_geometric
from torch_geometric.datasets import Flickr

def SkipGram(phi, walk, win_size, optimizer):
    for v_j, j in enumerate(walk):
        for u_k in walk[j - win_size : j + win_size]:
            # Calculate loss:
            log_liklihood = None
            # Update optimizer. TODO: implement from scratch
            optimizer.step()
        

class DeepWalk:
    def __init__(self, G, win_size, embedding_size, walks_per_vertex, walk_length, seed=36) -> None:
        '''
        Core DeepWalk class. Contains main algorithm from paper including SkipGram.
        Inputs:
            - G (): UNSURE OF GRAPH TYPE YET.   
            - win_size (int): window size
            - embedding_size (int): embedding size of output representation
            - walks_per_vertex (int): number of walks per vertex
            - walk_length (int): random walk length
            - seed (int): random seed. default set to 36.
        '''
        self.G = G
        self.vertices = self.G.vertices()
        self.win_size = win_size
        self.embedding_size = embedding_size
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        self.seed = seed

    def _init_vertex_representations(self):
        '''Initialize phi - or embedding matrix - to be optimized.'''
        torch_geometric.seed_everything(self.seed) 
        self.phi = torch.rand(
            size = (len(self.G), self.embedding_size)
        )

    def _construct_binary_tree(self):
        '''I feel like this is just an adjacancy matrix.'''
        pass

    def calculate_embeddings(self):
        '''Main function run to calculate embedding matrix.'''
        self._init_vertex_representations()
        binary_tree = self._construct_binary_tree() 

        # Main loop in question:
        for walk_num in range(len(self.walks_per_vertex)):
            # Shuffle V
            # O = shuffle(V)
            for vert_idx in range(len(O)):
                walk = self.random_walk(self.G, self.vertices[vert_idx], self.walk_length)
                SkipGram(self.phi, walk, self.win_size)