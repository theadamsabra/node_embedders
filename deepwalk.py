import torch
import torch_geometric
from random_walk import RandomWalk
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
        self.random_walk = RandomWalk(self.walks_per_vertex, self.walk_length)
        self.seed = seed

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
        self._init_vertex_representations()

        # Main loop in question:
        for walk_num in range(0, self.walks_per_vertex):
            # Shuffle V
            O = self._shuffle()
            # Now walk through each vertex and update weights
            for vertex in O:
                vertex = vertex.item() 
                walk = self.random_walk(self.edges, vertex)
