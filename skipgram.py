import torch

class SkipGram:
    '''
    Skipgram implementation from:
        https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
    '''
    def __init__(self) -> None:
        pass
    
    def __call__(self, phi, walk, win_size, tree) -> torch.Tensor:
        '''
        Core SkipGram algorithm.
        Inputs:
            - phi (torch.nn.Embedding): embedding matrix of size (num_vertices, embedding_dim). This will be the final embedding matrix.
            - walk (list): random walk path.
            - win_size (int): window size.
            - tree (binary_tree.Tree object): binary tree representation of graph.
        '''
        for j, v_j in enumerate(walk):

            for u_k in walk[j-win_size:j+win_size]:

                # Factor out u_k and get probability:
                nodes = tree.getNodeList(u_k) 

                print(u_k)
        
        return phi 