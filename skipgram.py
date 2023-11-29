import torch

class SkipGram:
    '''
    Skipgram implementation from:
        https://papers.nips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
    Borrowed from https://github.com/loginaway/DeepWalk/blob/master/deepwalk/SkipGram.py as the Tree class also borrows from it.
    To try to learn while not doing this entirely in the dark, I made it a torch based implementation.
    '''
    def __init__(self, phi, tree) -> None:
        self.phi = phi
        self.tree = tree
        self.losses = [] 

    def __call__(self, walk, win_size) -> torch.Tensor:
        '''
        Core SkipGram algorithm.
        Inputs:
            - phi (torch.nn.Embedding): embedding matrix of size (num_vertices, embedding_dim). This will be the final embedding matrix.
            - walk (list): random walk path.
            - win_size (int): window size.
        '''
        for j, v_j in enumerate(walk):

            for u_k in walk[j-win_size:j+win_size+1]:
                # Get nodes and their Phis:
                nodes, Phis = self.get_nodes_and_phis(u_k)
                # Get loss and save:
                loss = self.calculate_loss(nodes, v_j)
                self.losses.append(loss)
                # Calculate gradient 
                self.phi = self.sgd(v_j, nodes, Phis) 

        return self.phi

    def calculate_loss(self, nodes, v_j):
        '''
        Calculate loss from nodes and vertex.
        '''
        return

    def sgd(self, nodes, Phis):
        '''
        Optimize given nodes and Phis.
        ''' 

    def get_nodes_and_phis(self, u_k):
        '''
        Get nodes from root to u_k and every leaf's corresponding Phi.
        '''
        nodes = self.tree.getNodeList(u_k)
        Phis = torch.empty(
            size=(len(nodes), len(nodes[0].Phi))
        )
        for i in range(len(nodes)):
            Phis[i] = nodes[i].Phi.flatten()

        return nodes, Phis
        