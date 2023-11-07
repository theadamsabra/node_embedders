import torch
import torch_geometric
from torch import optim
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

    def _init_vertex_representations(self):
        '''Initialize phi - or embedding matrix - to be optimized.'''
        torch_geometric.seed_everything(self.seed) 
        self.phi = torch.rand(
            size = (self.num_vertices, self.embedding_size)
        )
        # From Section 4.2.3 in paper
        self.optimizer = optim.SGD(params=self.phi, lr=0.025)

    def _construct_binary_tree(self):
        '''I feel like this is just an adjacancy matrix.'''
        pass
    
    def _shuffle(self):
        '''Shuffle vertices'''
        return torch.randperm(self.num_vertices) 

    def calculate_embeddings(self):
        '''Main function run to calculate embedding matrix.'''
        self._init_vertex_representations()
        binary_tree = self._construct_binary_tree() 

        # Main loop in question:
        for walk_num in range(len(self.walks_per_vertex)):
            # Shuffle V
            O = self.shuffle()
            for vert_idx in range(len(O)):
                walk = self.random_walk(self.G, self.vertices[vert_idx], self.walk_length)
                SkipGram(self.phi, walk, self.win_size)

# Used for debugging for now:
if __name__ == "__main__":
    G = Flickr('data/flickr')
    optimizer = optim.SGD()
    deepwalk = DeepWalk(G, 2, 1024, 10, 5, optimizer)
    deepwalk.calculate_embeddings()