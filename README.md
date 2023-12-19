# Node Embedders

Unofficial implementation of [Deepwalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf). The ultimate objective of this repository is to implement a torch-geometric friendly node embedder tool.


## Usage

First, setup the python environment:

```bash
conda create -n node-embedders
conda activate node-embedders
pip install -r requirements.txt
```

Then, to calculate embeddings, leveraging the template code is sufficient.

```python
from node_embedder import NodeEmbedder 
from torch_geometric.datasets import Flickr

G = Flickr(root='path/to/dataset') # G can be any dataset from torch_geometric.datasets
win_size = 10
embedding_size = 128 
walks_per_vertex = 80
walk_len = 40
num_workers = 16
percent_data = 0.01
deepwalk = NodeEmbedder(G, win_size, embedding_size, walks_per_vertex, walk_len, num_workers, percent_data)
deepwalk.calculate_embeddings()
```

After calculating the embeddings, you can access which vertices have been used as well as their embeddings using the following:

```python
trained_vertices = deepwalk.trained_vertices # list containing all unique vertices seen in training (in order of node index.) 
trained_weights = deepwalk.trained_weights # weights of each vertex in trained_vertices (in same order of trained_vertices.)
```
