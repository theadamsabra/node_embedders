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

    def random_walk(self, edges, vertex):
        '''
        Core random walk algorithm.
        Inputs:
            - edges (torch.Tensor): tensor of edge list.   
            - vertex (int): vertex to begin walk from.
        '''
        # walk_len = 0
        # while walk_len < self.walk_length
        # Get all connected nodes from vertex

        # Random choose

        # walk_len += 1
        pass