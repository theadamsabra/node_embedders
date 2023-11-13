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
            phi_vk = self._get_embedding(phi, v_j)

            for u_k in walk[j-win_size:j+win_size]:
                # Factor out u_k and get probability:
                phi_uk = self._get_embedding(phi, u_k) 

                print(u_k)
        
        return phi
    
    def _get_embedding(self, phi, vertex):
        '''
        Get embedding of vertex.
        Inputs:
            - phi (torch.nn.Embedding): embedding matrix of size (num_vertices, embedding_dim)
            - vertex (int): vertex number
        Outputs:
            - vertex_embedding (torch.Tensor): embedding representation of vertex.
        '''
        vertex_tensor = torch.Tensor([vertex]).type(torch.long)
        return phi(vertex_tensor)