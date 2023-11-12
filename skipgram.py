import torch

class SkipGram:
    '''
    Skipgram implementation from:
        https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
    '''
    def __init__(self) -> None:
        pass
    
    def __call__(self, phi, walk, win_size) -> torch.Tensor:
        '''
        Core SkipGram algorithm.
        Inputs:
            - phi (torch.nn.Embedding): embedding matrix of size (num_vertices, embedding_dim)
            - walk (list): random walk path.
            - win_size (int): window size.
        '''
        for j, v_j in enumerate(walk):
            # Get embedding of node
            vertex = torch.Tensor(v_j).type(torch.long)
            phi_j = phi(vertex)

            for u_k in walk[j-win_size:j+win_size]:
                pass
        
        return phi