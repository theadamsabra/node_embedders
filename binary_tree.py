# Copied over and updated for my use case from:
# https://github.com/YCAyca/Data-Structures-and-Algorithms-with-Python/blob/main/Huffman_Encoding/huffman.py

import multiprocessing

# A Huffman Tree Node
class Node:
    def __init__( prob, symbol, left=None, right=None):
        # probability of symbol
        prob = prob

        # symbol 
        symbol = symbol

        # left node
        left = left

        # right node
        right = right

        # tree direction (0/1)
        code = ''
        
def update_symbols(symbols, element):
    '''
    Update symbols of elements.

    Inputs:
        - symbols 
        - element: one of the two vertices connected in edge.
    '''
    if symbols.get(element) == None:
        symbols[element] = 1
    else:
        symbols[element] += 1

def calculate_probability(edges):
    '''
    Calculate probabilites of edges. Because this is used in a map function,
    we simply map over one of the rows twice. Thus, we treat it as a 1D array.
    
    Inputs:
        - edges (torch.Tensor): edge tensor of shape (2, num_edges)
    '''
    symbols = dict()
    for i in range(edges.shape[-1]):
        # Get elements and update symbols
        element_1 = edges[i].item()
        update_symbols(symbols, element_1)

        element_2 = edges[i].item()
        update_symbols(symbols, element_2)

    return symbols

def calculate_codes(node, val=''):
    codes = dict()

    new_value = val + str(node.code)
    if node.left:
        calculate_codes(node.left, new_value)
    if node.right:
        calculate_codes(node.right, new_value)
    if (not node.left and not node.right):
        codes[node.symbol] = new_value

    return codes

def output_encoded(data, coding):
    encoding_output = [coding[c] for c in data]
    string = ''.join([str(item) for item in encoding_output])
    return string

def merge_nodes(nodes):
    '''
    Merge nodes function
    '''
    while len(nodes) > 1:
        # Sort nodes
        nodes = sorted(nodes, key=lambda x: x.prob)

        # get 2 smallest nodes:
        right = nodes[0]
        left = nodes[1]

        left.code = -1
        right.code = 1

        # combine 2 smallest nodes to create new node:
        new_node = Node(left.prob + right.prob, left.symbol+right.symbol, left, right)
        nodes.remove(left)
        nodes.remove(right)
        nodes.append(new_node)

def construct_huffman_tree(edges):
    pool = multiprocessing.Pool()
    # Get probabilites:
    symbols = pool.map(calculate_probability, edges)
    # Update symbols to be one mapping:

    # Get symbols to construct nodes:        
    symbs = symbols.keys()
    nodes = []
    for symbol in symbs:
        nodes.append(Node(symbols.get(symbol), symbol))

    # Merge nodes: 
    pool.map(merge_nodes)

    # Get encoding and return it:
    huffman_encoding = calculate_codes(nodes[0])
    return huffman_encoding