import torch
import random
class RandomWalk:
    def __init__(self, walks_per_vertex, walk_length) -> None:
        '''
        Random walk algorithm. For now, will only work on undirected graphs.
        Inputs:
            - walks_per_vertex (int): number of walks per vertex
            - walk_length (int): random walk length
        '''
        self.walks_per_vertex = walks_per_vertex
        self.walk_length = walk_length 

    def __call__(self, edge_list, vertex):
        '''
        Core random walk algorithm.
        Inputs:
            - edge_list (torch.Tensor): tensor of edge list.   
            - vertex (int): vertex to begin walk from.
        '''
        # Instantiate walk path
        final_walk = [vertex]

        while len(final_walk) < self.walk_length:
            # Get all connected nodes from vertex
            next_vertices_array = self.find_edges(edge_list, vertex)
            # Random choose
            vertex = random.choice(next_vertices_array).item() 
            final_walk.append(vertex) 

        return final_walk

    def find_edges(self, edge_list, vertex):
        '''
        Find edges of vertex in edge_list. Assume undirected graph.
        Inputs:
            - edge_list (torch.Tensor): tensor of edge list 
            - vertex (int): vertex to begin walk from.
        '''
        # Edge list is [2 x num_connections]
        # dim1 will be a 0 or 1 indicating on which row we find vertex in.
        # dim2 will be the index of the column. 
        dim1, dim2 = (edge_list == vertex).nonzero(as_tuple=True)

        # This means if we want to find all connections of vertex,
        # we simply find the opposite of dim1 of the column of dim2:
        path = edge_list[~dim1, dim2]
        return path