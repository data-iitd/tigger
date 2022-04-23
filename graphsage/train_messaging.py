
import networkx as nx
import scipy.sparse as sp
import numpy as np
import utils_graphsage_1 as utils
import torch
import torch
from collections import defaultdict
import numpy as np
import time
import json
import pandas as pd

seed=114514
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
data= pd.read_csv("../data/opsahl-ucsocial/data.csv")
data = data.drop_duplicates(subset=['start','end'])
node_to_id = dict()
node_type_dict = dict()
for node in list(data['start']) + list(data['end']):
    node = int(node)
    if node not in node_to_id:
        node_to_id[node] = len(node_to_id)
for node in list(data['start']):
    node = node_to_id[int(node)]
    node_type_dict[node] = 1 ### user type node
for node in list(data['end']):
    node = node_to_id[int(node)]
    node_type_dict[node] = 0 ### item type node

#edges = defaultdict(lambda: 0)
train_ones = []
feature_dict = {}
for start,end in data[['start','end']].values:
    start,end = int(start),int(end)
    start = node_to_id[start]
    end = node_to_id[end]
    train_ones.append([int(start),int(end)])
train_ones= np.array(train_ones)
print(len(train_ones))
print(train_ones[:5])
### There can be multiple edges between same node  pairs


adj_sparse = np.zeros((np.max(train_ones)+1,np.max(train_ones)+1))
for e in train_ones:
    adj_sparse[e[0],e[1]]=1
    adj_sparse[e[1],e[0]]=1
    
adj_sparse = sp.coo_matrix(adj_sparse).tocsr()

# lcc = utils.largest_connected_components(adj_sparse)
# adj_sparse= adj_sparse[lcc,:][:,lcc]
_N = adj_sparse.shape[0]
print('n',_N)
_Edges=[]
for x in np.column_stack(adj_sparse.nonzero()):
    if not x[0]==x[1]:
        _Edges.append((x[0],x[1]))
_num_of_edges=int(len(_Edges)/2)
print('m',_num_of_edges)

dic=defaultdict(set)
for x in _Edges:
    a1=x[0]
    a2=x[1]
    dic[a1].add(a2)
    dic[a2].add(a1)
    

adj_origin=np.zeros((_N,_N)).astype(int)  ### extra dimension for node type
for (i,j) in _Edges:
    adj_origin[i][j]=1
    adj_origin[j][i]=1
assert(np.sum(adj_origin==adj_origin.T)==_N*_N)
assert(np.sum(adj_origin)==_num_of_edges*2)
# feature_matrix = np.zeros((adj_origin.shape[0], adj_origin.shape[1]))
# feature_matrix[:,:] = adj_origin 
# for i in range(_N):
#     feature_matrix[i][i] = 1  ### one hot encoding
#replace feat_matrix = adj_origin with feat_matrix = feature_matrix in below initialization
embedding_dim=128

graphsagemodel=utils.GraphSAGE(_N=_N,_M=_num_of_edges,adj_origin=adj_origin,feat_matrix = adj_origin,
                                         adj_dic=dic,embedding_dim=embedding_dim)


graphsagemodel.graphsage_train(boost_times=100,add_edges=1000,training_epoch=20000,
                               boost_epoch=4000,learning_rate=0.001,save_number=0,dirs='models/ucsocial')
#graphsagemodel.save_model(path='graphsage_model_node_type/graph_graphsage.pth',embedding_path='graphsage_model_node_type/embeddings.npy')
