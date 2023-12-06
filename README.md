# Deepwalk Implementation

Unofficial implementaion of [Deepwalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf).


## Usage

First, setup your python environment:

```bash
conda create -n deepwalk
conda activate deepwalk
pip install -r requirements.txt
```

Then, to calculate embeddings, leveraging the template code is sufficient.

```python
from deepwalk import DeepWalk
from torch_geometric.datasets import Flickr

G = Flickr() # G can be any dataset from torch_geometric.datasets
win_size = 10
embedding_size = 128 
walks_per_vertex = 80
walk_len = 40
num_workers = 16
percent_data = 0.01
deepwalk = DeepWalk(G, win_size, embedding_size, walks_per_vertex, walk_len, num_workers, percent_data)
deepwalk.calculate_embeddings()
```

After calculating the embeddings, you can access which vertices have been used as well as their embeddings using the following:

```python
trained_vertices = deepwalk.trained_vertices # list containing all unique vertices seen in training 

trained_weights = deepwalk.trained_weights # weights of each vertex in trained_vertices (in order.)
```