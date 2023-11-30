import torch
import math
from concurrent.futures import ProcessPoolExecutor
from gensim.models import Word2Vec
from random_walk import RandomWalk
from torch_geometric.seed import seed_everything
from torch_geometric.datasets import Flickr

class DeepWalk:
    def __init__(self, G, win_size, embedding_size, walks_per_vertex, \
                 walk_length, num_workers, percent_data, seed=36) -> None:
        '''
        Core DeepWalk class. 
        Inputs:
            - G (torch_geometric.data.Data): torch geometric data object.   
            - win_size (int): window size
            - embedding_size (int): embedding size of output representation
            - walks_per_vertex (int): number of walks per vertex
            - walk_length (int): random walk length
            - num_workers (int): number of CPU workers for SkipGram.
            - percent_data (float): percentage of data to run algorithm on.
            - seed (int): random seed. default set to 36.
        '''
        self.G = G
        self.num_vertices = self.G[0].num_nodes 
        self.edges = self.G.edge_index
        self.win_size = win_size
        self.embedding_size = embedding_size
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length
        self.seed = seed
        self.random_walk = RandomWalk(self.walks_per_vertex, self.walk_length)
        self.num_workers = num_workers
        self.percent_data = percent_data
        self.skipgram = Word2Vec(
            vector_size=self.embedding_size,
            window=self.win_size,
            workers=self.num_workers,
            sg=1, # Default sg to 1 to leverage SkipGram and not CBOW
            hs=1 # Default hs to 1 to leverage Hierarchical Softmax
        )

    def _shuffle(self, num_vertices):
        '''Shuffle vertices'''
        return torch.randperm(num_vertices) 

    def random_walk_vertex(self, vertex):
        vertex = vertex.item()
        walk = self.random_walk(self.edges, vertex)
        return walk

    def construct_all_walks(self, sampled_nodes):
        '''
        Construct all random walks and save as method.
        
        Inputs:
            - sampled_nodes (torch.Tensor):
        '''
        # Number of walks per vertex
        all_walks = []

        for _ in range(0, self.walks_per_vertex):
            # Shuffle V
            O = sampled_nodes[self._shuffle(len(sampled_nodes))]
            # Get all walks
            walks = [self.random_walk_vertex(vertex) for vertex in O]
            all_walks.append(walks)

        return all_walks 

    def sample_from_graph(self) -> torch.Tensor:
        '''
        Update attributes as subsample of graph.
        '''
        # Randomly select the data we sample
        num_sampled = math.ceil(self.num_vertices * self.percent_data)
        shuffled_indices = self._shuffle(self.num_vertices)
        sampled_nodes = shuffled_indices[:num_sampled]
        return sampled_nodes

    def calculate_embeddings(self):
        # Seed everything for reproducibility
        seed_everything(self.seed) 

        # Sample from graph
        sampled_nodes = self.sample_from_graph()

        # Construct all walks and run it in parallel:
        all_walks = self.construct_all_walks(sampled_nodes)
        pass

# Used for debugging for now:
if __name__ == "__main__":
    G = Flickr('data/flickr')
    win_size = 10
    embedding_size = 128 
    walks_per_vertex = 80
    walk_len = 40
    num_workers = 16
    percent_data = 0.01
    deepwalk = DeepWalk(G, win_size, embedding_size, walks_per_vertex, walk_len, num_workers, percent_data)
    deepwalk.calculate_embeddings()