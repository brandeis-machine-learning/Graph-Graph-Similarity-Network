# Graph-Graph-Similarity-Network
This repository contains the Python implementation for the paper "Graph-Graph Similarity Network".


## Paper Abstract

Graph learning aims to predict the label for an entire graph. Recently, Graph Neural Networks (GNNs)-based approaches become an essential strand to learning low-dimensional continuous embeddings of entire graphs for graph label prediction. While GNNs explicitly aggregate the neighborhood information and implicitly capture the topological structure for graph representation, they ignore the relationships among graphs. In this paper, we propose a Graph-Graph Similarity Network to tackle the graph learning problem by constructing a SuperGraph through learning the relationships among graphs. Each node in the SuperGraph represents an input graph, and the weights of edges denote the similarity between graphs. By this means, the graph learning task is then transformed into a classical node label propagation problem. Specifically, we employ an Adversarial Autoencoder to align embeddings of all the graphs to a prior data distribution. After the alignment, we design the Graph-Graph Similarity Network to learn the similarity between graphs, which functions as the adjacency matrix of the SuperGraph. By running node label propagation algorithms on the SuperGraph, we can predict the labels of graphs. Experiments on five widely used classification benchmarks and four public regression benchmarks under a fair setting demonstrate the effectiveness of our method.


## File Description

* `datasets.py`: Data loading and triplet generation
* `layers.py`: GraphConvolution Layer
* `model.py`: Pytorch implementation of the G2G model
* `train_batch.py`: Model training


## How to Run

Train the model by running:
```bash
python train_batch.py
```
