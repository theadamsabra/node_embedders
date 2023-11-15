import torch


class BinaryTree:
    def __init__(self) -> None:
       self.node_count = {} 

    def get_subtrees(self, tree):
        yield tree
        for subtree in tree:
            if type(subtree) == list:
                for sub_subtree in self.get_subtrees(subtree):
                    yield sub_subtree

    def generate_paths_of_leaves(self, tree):
        '''
        Generate the paths of leaves in the tree.
        '''
        for i, subtree in enumerate(tree):
            if type(subtree) == list:
                # No shot python will support this reliably
                for path, value in self.generate_paths_of_leaves(subtree): 
                    yield [i] + path, value
            else:
                yield [i], subtree

    def count_nodes(self, tree):
        if id(tree) in self.node_count:
            return self.node_count[id(tree)]

        size = 0
        for node in tree:
            if type(node) == list:
                size += 1 + self.count_nodes(node)               
        
        self.node_count[id(self.node_count)] = size
        return size
    
    def get_nodes(self, tree, path):
        next_node = 0
        nodes = []
        for decision in path:
            nodes.append(next_node)
            next_node += 1 + self.count_nodes(tree[:decision])
        
    def construct_binary_tree(self, vertices) -> list:
        '''
        Construct binary tree from vertices.

        Inputs:
            - vertices (torch.Tensor): flat tensor of shape (num_vertices, 1). Used
            to construct the binary tree.

        Outputs:
            - binary_tree (list): binary tree output.
        ''' 
        while len(vertices) > 2:

            tmp_outputs = []

            for i in range(0, len(vertices), 2):
                if len(vertices) - (i+1) > 0:
                    tmp_outputs.append([vertices[i], vertices[i+1]])
                else:
                    tmp_outputs.append(vertices[i])
            vertices = tmp_outputs

        return vertices