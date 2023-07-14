#%%
import os
import networkx as nx
import scipy.sparse as sp
import numpy as np
import utils_graphsage_1 as utils
import utils_graphsage_1
import torch
import torch
from collections import defaultdict
import numpy as np
import pandas as pd
#%%
seed=114514
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.seed_all()

# # create dummy feature matrixs
# node_df = np.random.random(size=(1899, 10))
# pdf = pd.DataFrame(node_df, columns = ["attr"+str(i) for i in range(10)])
# pdf['id']=range(1899)
# pdf.to_parquet("../data/opsahl-ucsocial/node_features.parquet")

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
data= pd.read_csv("../data/opsahl-ucsocial/data.csv")
data = data.drop_duplicates(subset=['start','end'])
features = pd.read_parquet("../data/opsahl-ucsocial/node_features.parquet")


#%% create mapping dict from node to id
node_to_id = dict()  # dictionary with node ids
def get_id(node, node_to_id):
    """check if node is in node_to_id dict and adds when missing"""
    node = int(node)
    if node not in node_to_id:
        node_id = len(node_to_id)
        node_to_id[node] = node_id
        return node_id
    else:
        return node_to_id[node]

#%% Built dict containing a set of neighsbors per node id. 
neighbors_dict = defaultdict(set) 
for start, end in data[['start','end']].values:
    start_id = get_id(start, node_to_id)
    end_id = get_id(end, node_to_id)
    neighbors_dict[start_id].add(end_id)
    neighbors_dict[end_id].add(start_id)


#%% prep feature matrix
node_mapping_df = pd.DataFrame.from_dict(node_to_id, orient='index', columns=['new_id'])
node_mapping_df = node_mapping_df.reset_index(names='id')
features = features.merge(node_mapping_df, on='id', how='inner')
features = features.drop(['id', 'new_id'], axis=1)

_N = features.shape[0]
_M = data.shape[0]
assert _N==len(node_to_id), "N is different from the number of node id's"


#%%
embedding_dim=128
epoch = 200  #20000  # epoch for training
boost_epoch=400 #4000 # epoch per boosting run
boost_times = 3 #100 # number of boosting runs
import time
# import utils_graphsage_1
import importlib
importlib.reload(utils_graphsage_1)

graphsagemodel=utils_graphsage_1.GraphSAGE(_N=_N,
                               feat_matrix = features.values,
                               adj_dic=neighbors_dict,
                               embedding_dim=embedding_dim,
                               verbose_level=2)

start = time.process_time_ns()
graphsagemodel.graphsage_train(boost_times=boost_times,  # number of boosting runs.
                               add_edges=1000,  # number of edges added during boosting
                               training_epoch=epoch,  # epoch for normal training
                               boost_epoch=boost_epoch,  # epoch for boosting with added edges.
                               learning_rate=0.001,
                               save_number=0,dirs='models/ucsocial')
end = time.process_time_ns()
print(f"duration {(end-start)/1e9} sec")
#graphsagemodel.save_model(path='graphsage_model_node_type/graph_graphsage.pth',embedding_path='graphsage_model_node_type/embeddings.npy')

# %%
