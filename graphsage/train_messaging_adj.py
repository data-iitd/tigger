#%%
import torch
import time
import utils_graphsage_1
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def get_id(node, node_to_id):
        """check if node is in node_to_id dict and adds when missing"""
        node = int(node)
        if node not in node_to_id:
            node_id = len(node_to_id)
            node_to_id[node] = node_id
            return node_id
        else:
            return node_to_id[node]

def prep_input_data(data, features, max_degree=10000):
    """trains graphsage and stores trained model with embedding in the output path"""
    node_to_id = dict()  # dictionary with node ids
    
    # Built dict containing a set of neighsbors per node id. 
    neighbors_dict = defaultdict(set) 
    print("creating neighbors dicts")
    for start, end in tqdm(data[['start','end']].values):
        start_id = get_id(start, node_to_id)
        end_id = get_id(end, node_to_id)
        if len(neighbors_dict[start_id])<max_degree and len(neighbors_dict[end_id])<max_degree:
            neighbors_dict[start_id].add(end_id)
            neighbors_dict[end_id].add(start_id)

    # prep feature matrix
    node_mapping_df = pd.DataFrame.from_dict(node_to_id, orient='index', columns=['new_id'])
    node_mapping_df = node_mapping_df.reset_index(names='id')
    features = features.merge(node_mapping_df, on='id', how='inner')
    features = features.drop(['id', 'new_id'], axis=1)

    _N = features.shape[0]
    _M = data.shape[0]
    assert _N==len(node_to_id), "N is different from the number of node id's"
    
    return ({'_N': _N, 'feat_matrix': features.values, 'adj_dic': neighbors_dict},
            {'node_to_id': node_to_id})
    

def train_and_calculate_graphsage_embedding(init_dict, train_dict):
    graphsagemodel=utils_graphsage_1.GraphSAGE(**init_dict)
    
    start = time.process_time_ns()
    train_metrics = graphsagemodel.graphsage_train(**train_dict)
    end = time.process_time_ns()
    print(f"duration {(end-start)/1e9} sec")
    graphsagemodel.save_model(train_dict['dirs']+'/model_final')
    graphsagemodel.save_embedding(train_dict['dirs']+'/embedding_matrix_final')
    
    return train_metrics
    
    
def print_metrics(train_metrics):
  epoch_cnt = 0
  for metrics in train_metrics:
      epoch =  [x+epoch_cnt for x in metrics['epoch']]
      plt.plot(epoch, metrics['train_loss'], label=metrics['label'] )

      if 'val_loss' in metrics.keys():
          epoch =  [x+epoch_cnt for x in metrics['val_epoch']]
          plt.plot(epoch, metrics['val_loss'], label=metrics['label']+"_val" )
          
      epoch_cnt = epoch_cnt + max(metrics['epoch'])/ (len(metrics['epoch']) - 1 ) * len (metrics['epoch'])

  plt.legend(bbox_to_anchor=(1.10, 1))
  plt.show()

# %%
# calculate embedding
if __name__ == "__main__":
    output_path = 'models/ucsocial'
    data = pd.read_csv("../data/opsahl-ucsocial/data.csv")
    data = data.drop_duplicates(subset=['start','end'])
    features = pd.read_parquet("../data/opsahl-ucsocial/node_features.parquet")
    
    # features = pd.read_parquet("../data_large/5mln_node_features.parquet")
    # data = pd.read_parquet("../data_large/5mln_edge.parquet")

    init_dict1 = {
        'embedding_dim': 128,
        'verbose_level': 2
    }

    train_dict = {
        'training_epoch': 400,  #20000  # epoch for training
        'boost_epoch': 400, #4000 # epoch per boosting run
        'boost_times': 0, #100 # number of boosting runs
        'add_edges': 10, 
        'learning_rate': 0.0001,
        'save_number': 0,
        'dirs': output_path
    }

    init_dict2, info_dict = prep_input_data(data, features)
    train_metrics = train_and_calculate_graphsage_embedding({**init_dict1, **init_dict2}, train_dict)
    print_metrics(train_metrics)
    # %%


def embedding_to_pandas(embedding, node_to_id):
    """calculates the node embedding and adds them in pandas df with original ID"""
    df = pd.DataFrame(embedding)
    ids = dict((v,k) for k,v in node_to_id.items())
    id_df = pd.DataFrame.from_dict(ids, orient='index', column='id')
    embed_df = id_df.join(df, how='outer')
    return embed_df
