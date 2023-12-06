import torch
import math
import multiprocessing as mp
from gensim.models import Word2Vec
from random_walk import RandomWalk
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
        self.random_walk = RandomWalk(self.walks_per_vertex, self.walk_length)
        self.num_workers = num_workers
        self.percent_data = percent_data
        self.skipgram = Word2Vec(
            vector_size=self.embedding_size,
            window=self.win_size,
            workers=self.num_workers,
            sg=1, # Default sg to 1 to leverage SkipGram and not CBOW
            hs=1, # Default hs to 1 to leverage Hierarchical Softmax
            min_count=1
        )

    def _shuffle(self, num_vertices):
        '''Shuffle vertices'''
        return torch.randperm(num_vertices) 

    def construct_all_walks(self, O):
        '''
        Construct all random walks.
        
        Inputs:
            - O (torch.Tensor): shuffled vertices.
        '''
        # Get all walks:
        walks = [self.random_walk(self.edges, vertex.item()) for vertex in O]
        return walks 

    def sample_from_graph(self) -> torch.Tensor:
        '''
        Update attributes as subsample of graph.
        
        Outputs:
            - sampled_nodes (torch.Tensor): sampled nodes from graph.
        '''
        # Randomly select the data we sample
        num_sampled = math.ceil(self.num_vertices * self.percent_data)
        shuffled_indices = self._shuffle(self.num_vertices)
        sampled_nodes = shuffled_indices[:num_sampled]
        return sampled_nodes

    def construct_accesible_weights(self):
        '''
        Parse out relevant information for easily accessible weights.
        '''
        self.trained_vertices = self.skipgram.wv.index_to_key
        self.trained_weights = self.skipgram.syn1

    def calculate_embeddings(self):
        # Sample from graph:
        sampled_nodes = self.sample_from_graph()

        # Shuffle vertices and construct all walks:
        for i in range(self.walks_per_vertex):
            O = sampled_nodes[self._shuffle(len(sampled_nodes))]
            all_walks = self.construct_all_walks(O)

            update = False if i == 0 else True 
            self.skipgram.build_vocab(corpus_iterable=all_walks, update=update)

            self.skipgram.train(
                corpus_iterable=all_walks,
                total_examples=self.skipgram.corpus_count,
                epochs=1 # doesn't default automatically for some reason
            )
        
        # Once we're good with training, make everything easily accessible 
        self.construct_accesible_weights()