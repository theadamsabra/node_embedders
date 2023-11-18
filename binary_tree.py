import torch
from copy import copy

class BinaryTree:
    def __init__(self, leaf_num:int) -> None:
        self.decision = {0:1, 1:-1}
        self.index = 0

        arr = []
        arr2 = []

        for path, base, decision, leaf in self.get_decision(self.construct_binary_tree(leaf_num)):
            tmp = torch.zeros((1, leaf_num-1)).type(torch.long) 
            tmp2 = torch.ones((1, leaf_num-1)).type(torch.long)

            tmp.put(
                torch.Tensor(path).type(torch.long), 
                torch.Tensor(decision).type(torch.long)
            )

            tmp2.put(
                torch.Tensor(path).type(torch.long), 
                torch.Tensor(base).type(torch.long)
            )

            arr.append(tmp)
            arr2.append(tmp2)
        
        self.index = 0
        self.decision = torch.Tensor(arr).to(torch.long)
        self.base = torch.Tensor(arr2).to(torch.long)

    def get_decision(self, tree):
        tmp_i = self.index
        self.index += 1

        for i, subtree in enumerate(tree):
            if type(subtree) == list:
                for path, base, decision_list, value in self.get_decision(subtree):
                    yield [tmp_i] + path, [i]+base, [self.decision[i]]+decision_list, value
            else:
                yield [tmp_i], [i], [self.decision[i]],subtree                     

    def construct_binary_tree(self, vertices) -> list:
        '''
        Construct binary tree from vertices.

        Inputs:
            - vertices (torch.Tensor): flat tensor of shape (num_vertices, 1). Used
            to construct the binary tree.

        Outputs:
            - binary_tree (list): binary tree output.
        ''' 
        vertices = list(range(vertices))
        vertices = copy(vertices)
        while len(vertices) > 2:

            tmp_outputs = []

            for i in range(0, len(vertices), 2):
                if len(vertices) - (i+1) > 0:
                    tmp_outputs.append([vertices[i], vertices[i+1]])
                else:
                    tmp_outputs.append(vertices[i])
            vertices = tmp_outputs

        return vertices