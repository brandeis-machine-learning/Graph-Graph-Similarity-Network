import scipy.io as sp
import os
import numpy as np
from keras.utils import to_categorical

import networkx as nx
import scipy as sc
import re
import torch
import torch.utils.data
import random
import json


def encode2onehot(arr):
    mapC = {}
    for i in range(len(arr)):
        if arr[i] not in mapC.keys():
            mapC[arr[i]] = 1
    x = arr[:, np.newaxis]
    return (np.array(list(mapC.keys())==x[:])).astype(np.integer)


def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        pass
        # print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            #if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    #graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    #if label_has_zero:
    #    graph_labels += 1
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    for i in range(1,1+len(adj_list)):
    # for i in range(1,21):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
        if max_nodes > 0 and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
            elif len(node_labels) > 0:
                G.node[u]['feat'] = G.node[u]['label']
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]
        elif len(node_labels) > 0:
            G.graph['feat_dim'] = num_unique_node_labels
        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    return graphs





class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        self.adj_padded_all = []
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        
        self.assign_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        self.feat_dim = len(G_list[0].node[0]['feat'])


        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)


            num_nodes = adj.shape[0]
            adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
            adj_padded[:num_nodes, :num_nodes] = adj
            
            self.adj_padded_all.append(adj_padded)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = G.node[u]['feat']
                self.feature_all.append(f)
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))


            if assign_feat == 'id':
                self.assign_feat_all.append(
                        np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
            else:
                self.assign_feat_all.append(self.feature_all[-1])
            
        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        # use all nodes for aggregation (baseline)

        return {'adj':self.feature_all[idx].copy(),
                'feats':self.feature_all[idx].copy(),
                'label':self.label_all[idx].copy(),
                'num_nodes': num_nodes,
                'assign_feats':self.assign_feat_all[idx].copy()}








class Data_Loader():
    def __init__(self, root_path, dataset_name, train_ratio=0.9, batch_size=1, limit_nodes=0):
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.limit_nodes = limit_nodes
        self.graphs = read_graphfile(self.root_path, self.dataset_name, max_nodes=self.limit_nodes)
        self.max_num_nodes = max([G.number_of_nodes() for G in self.graphs])
        # self.k = k
        # self.data_train, self.data_test, self.adj_train, self.adj_test, self.label_train, self.label_test, self.domain_train, self.domain_test, self.source_name, self.label_name, self.nDims = self.__load_data__()
    
    def __load_data__(self, k=0):
        # source_name = set()
        label_name = set()
        # self.graphs = read_graphfile(self.root_path, self.dataset_name, max_nodes=self.limit_nodes)
        # self.max_num_nodes = max([G.number_of_nodes() for G in graphs])

        # # -------normal split-------------
        # train_idx = int(len(graphs) * self.train_ratio)
        # train_graphs = graphs[:train_idx]
        # test_graphs = graphs[train_idx:]

        # -------k-fold split-------------
        train_graphs, test_graphs = [], []

        if self.dataset_name == 'MUTAG':
            max_k = int(1 / (1 - self.train_ratio))
            for i in range(len(self.graphs)):
                if i % max_k == k:
                    test_graphs.append(self.graphs[i])
                else:
                    train_graphs.append(self.graphs[i])
        else:
            splits_filename = "data/data_splits/" + self.dataset_name + "_splits.json"
            splits = json.load(open(splits_filename, "r"))
            test_index = splits[k]['test']
            # print(k, "~~~~~~~~~~~~~~~~~", test_index)
            for i in range(len(self.graphs)):
                if i in test_index:
                    test_graphs.append(self.graphs[i])
                else:
                    train_graphs.append(self.graphs[i])

        # # -------random split-------------
        # train_idx = int(len(graphs) * self.train_ratio)
        # indices = np.arange(len(graphs))
        # np.random.shuffle(indices)
        # train_graphs, test_graphs = [], []
        # for i in range(len(graphs)):
        #     if i < train_idx:
        #         train_graphs.append(graphs[indices[i]])
        #     else:
        #         test_graphs.append(graphs[indices[i]])

        # minibatch
        dataset_sampler = GraphSampler(train_graphs, max_num_nodes=self.max_num_nodes, normalize=False)
        # for length in dataset_sampler.len_all:
        #     source_name.add(length)
        for label in dataset_sampler.label_all:
            label_name.add(label)

        label_train = dataset_sampler.label_all
        data_train = dataset_sampler.feature_all
        adj_train = dataset_sampler.adj_padded_all
        # domain_train = dataset_sampler.len_all

        dataset_sampler = GraphSampler(test_graphs, max_num_nodes=self.max_num_nodes, normalize=False)
        # for length in dataset_sampler.len_all:
        #     source_name.add(length)
        for label in dataset_sampler.label_all:
            label_name.add(label)

        label_test = dataset_sampler.label_all
        data_test = dataset_sampler.feature_all
        adj_test = dataset_sampler.adj_padded_all
        # domain_test = dataset_sampler.len_all

        labels = encode2onehot(np.concatenate((np.array(label_train), np.array(label_test)), axis=0))
        train_idx = len(label_train)
        self.data_train = data_train
        self.data_test = data_test
        self.adj_train = adj_train
        self.adj_test = adj_test
        self.label_train = labels[:train_idx]
        self.label_test = labels[train_idx:]
        self.label_name = label_name
        self.nDims = dataset_sampler.feat_dim




    def grmGenerator(self, model, batch_size=1, batch_sept=1, batch_index=0):
        trainSamples, trainAdjs, trainLabels = [], [], []
        for i in range(len(self.data_train)):
            if i % batch_sept == batch_index:
                trainSamples.append(self.data_train[i])
                trainAdjs.append(self.adj_train[i])
                trainLabels.append(self.label_train[i])

        trainSamples = np.array(trainSamples)
        trainAdjs = np.array(trainAdjs)

        model.eval()
        _, z, _, _, _, _ = model(trainSamples, trainAdjs)

        newSamples = z.detach().cpu().numpy()

        # indices = np.arange(len(newSamples))
        # np.random.shuffle(indices)
        # start = 0
        while True:
            # source_idx = indices[start * batch_size : (start + 1) * batch_size]
            # if len(source_idx) == 0:
            #     print(start, "ERROR!! Out of batch indices!!!!!")
            #     break
            # start += 1
            source_idx = np.random.choice(np.arange(len(newSamples)), size=len(newSamples), replace=False)
            source_data = newSamples[source_idx]
            sample_data = np.random.normal(0, 1, size=source_data.shape)

            batch_x, batch_a, batch_y = [], [], []
            for i in range(len(source_idx)):
                batch_x.append(trainSamples[source_idx[i]])
                batch_a.append(trainAdjs[source_idx[i]])
                batch_y.append(trainLabels[source_idx[i]])
            batch_x = np.array(batch_x)
            batch_a = np.array(batch_a)
            batch_y = np.array(batch_y)

            yield batch_x, batch_a, batch_y, source_data, sample_data

    def grmGenerator_no_similarity(self, model, batch_size=1, batch_sept=1, batch_index=0):
        trainSamples, trainAdjs, trainLabels = [], [], []
        for i in range(len(self.data_train)):
            if i % batch_sept == batch_index:
                trainSamples.append(self.data_train[i])
                trainAdjs.append(self.adj_train[i])
                trainLabels.append(self.label_train[i])

        trainSamples = np.array(trainSamples)
        trainAdjs = np.array(trainAdjs)

        model.eval()
        _, z, _, _, _ = model(trainSamples, trainAdjs)

        newSamples = z.detach().cpu().numpy()

        # indices = np.arange(len(newSamples))
        # np.random.shuffle(indices)
        # start = 0
        while True:
            # source_idx = indices[start * batch_size : (start + 1) * batch_size]
            # if len(source_idx) == 0:
            #     print(start, "ERROR!! Out of batch indices!!!!!")
            #     break
            # start += 1
            source_idx = np.random.choice(np.arange(len(newSamples)), size=len(newSamples), replace=False)
            source_data = newSamples[source_idx]
            sample_data = np.random.normal(0, 1, size=source_data.shape)

            batch_x, batch_a, batch_y = [], [], []
            for i in range(len(source_idx)):
                batch_x.append(trainSamples[source_idx[i]])
                batch_a.append(trainAdjs[source_idx[i]])
                batch_y.append(trainLabels[source_idx[i]])
            batch_x = np.array(batch_x)
            batch_a = np.array(batch_a)
            batch_y = np.array(batch_y)

            yield batch_x, batch_a, batch_y, source_data, sample_data


    def getTriplets(self, batch_y):
        label_y = np.argmax(batch_y, axis=1)
        num_classes = len(self.label_name)
        label_to_indices = {label: np.where(label_y == label)[0] for label in self.label_name}
        pairs = []
        n = min([len(label_to_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
        for d in range(num_classes):
            k_paris = []
            for i in range(n):
                z1, z2 = label_to_indices[d][i], label_to_indices[d][i + 1]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z3 = label_to_indices[dn][i]
                k_paris.append((z1, z2, z3))
            pairs.append(k_paris)
        return pairs