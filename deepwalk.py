import torch
import torch_geometric
from random_walk import RandomWalk
from binary_tree import Tree 
from skipgram import SkipGram
from torch_geometric.datasets import Flickr
from torch import nn
from torch import optim

class DeepWalk:
    def __init__(self, G, win_size, embedding_size, walks_per_vertex, walk_length, seed=36) -> None:
        '''
        Core DeepWalk class. 
        Inputs:
            - G (torch_geometric.data.Data): torch geometric data object.   
            - win_size (int): window size
            - embedding_size (int): embedding size of output representation
            - walks_per_vertex (int): number of walks per vertex
            - walk_length (int): random walk length
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
        self._init_vertex_representations()
        # Construct tree and grow it:
        self.tree = Tree(self.num_vertices, self.embedding_size) 
        self.tree.growTree()
        self.skipgram = SkipGram()

    def _init_vertex_representations(self):
        '''Initialize phi - or embedding matrix - to be optimized.'''
        torch_geometric.seed_everything(self.seed) 
        self.phi = nn.Embedding(
            num_embeddings=self.num_vertices,
            embedding_dim=self.embedding_size
        )
        self.phi.weight.data = torch.rand(
            size = (self.num_vertices, self.embedding_size)
        )
        # From Section 4.2.3 in paper
        self.optimizer = optim.SGD(params=self.phi.parameters(), lr=0.025)

    def _shuffle(self):
        '''Shuffle vertices'''
        return torch.randperm(self.num_vertices) 

    def calculate_embeddings(self):
        '''Main function run to calculate embedding matrix.'''
        # Main loop in question:
        for walk_num in range(0, self.walks_per_vertex):
            # Shuffle V
            O = self._shuffle()

            # Now walk through each vertex and update weights
            for vertex in O:
                # Get vertex in question and do random walk:
                vertex = vertex.item() 
                walk = self.random_walk(self.edges, vertex)
                # Update phi with skipgram: 
                self.phi = self.skipgram(self.phi, walk, self.win_size, self.tree)

# Used for debugging for now:
if __name__ == "__main__":
    G = Flickr('data/flickr')
    win_size = 10
    embedding_size = 128 
    walks_per_vertex = 80
    walk_len = 40
    deepwalk = DeepWalk(G, win_size, embedding_size, walks_per_vertex, walk_len)
    deepwalk.calculate_embeddings()