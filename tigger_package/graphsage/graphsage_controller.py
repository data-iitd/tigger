import time
import os
import pickle
from tqdm.auto import tqdm
from collections import defaultdict
import importlib
import tigger_package.graphsage.graphsage
importlib.reload(tigger_package.graphsage.graphsage) 
from tigger_package.graphsage.graphsage import GraphSAGE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class GraphSageController():
    def __init__(self, config_dict, path):
        self.config_path = path + config_dict['embed_path']
        for key, val in config_dict.items():
            setattr(self, key, val)
            
        self.node_to_id = dict() # dictionary with node ids
        os.makedirs(self.config_path, exist_ok=True)
        self.model_path = self.config_path + 'model/'
        os.makedirs(self.model_path, exist_ok=True)
        
    def get_embedding(self, nodes, edges):
        """trains graphsage and stores trained model with embedding in the output path"""
        edges = edges[['start', 'end']].drop_duplicates()
        
        init_dict1 = {
            'embedding_dim': self.embedding_dim,
            'verbose_level': self.verbose_level,
            'validation_fraction': self.validation_fraction,
            'batch_size': self.batch_size,
            'level': self.level,
            'dropout': self.dropout,
        }
        
        train_dict = {
            'training_epoch': self.training_epoch,
            'boost_epoch':self.boost_epoch,
            'boost_times': self.boost_times,
            'add_edges': self.add_edges,
            'learning_rate': self.learning_rate,
            'save_number': self.save_number,
            'dirs': self.model_path,
        }
        
        
        init_dict2 = self.prep_input_data(edges, nodes)
        train_metrics = self.train_and_calculate_graphsage_embedding({**init_dict1, **init_dict2}, train_dict)
        self.print_metrics(train_metrics)
        return train_metrics
        
    def prep_input_data(self, edges, nodes, max_degree=10000):
        """preps the adjancy and node features"""
        
        # Built dict containing a set of neighsbors per node id. 
        neighbors_dict = defaultdict(set) 
        print("creating neighbors dicts")
        for start, end in tqdm(edges.values):
            start_id = self.get_id(start)
            end_id = self.get_id(end)
            if len(neighbors_dict[start_id])<max_degree and len(neighbors_dict[end_id])<max_degree:
                neighbors_dict[start_id].add(end_id)
                neighbors_dict[end_id].add(start_id)

        # prep feature matrix
        node_mapping_df = pd.DataFrame.from_dict(self.node_to_id, orient='index', columns=['new_id'])
        nodes = nodes.merge(node_mapping_df, how='inner',  right_index=True, left_index=True)
        nodes = nodes.drop('new_id', axis=1)

        _N = nodes.shape[0]
        _M = edges.shape[0]
        assert _N==len(self.node_to_id), "N is different from the number of node id's"
        
        return {'_N': _N, 'feat_matrix': nodes.values, 'adj_dic': neighbors_dict}
            
    def get_id(self, node):
        """check if node is in node_to_id dict and adds when missing"""
        node = int(node)
        if node not in self.node_to_id:
            node_id = len(self.node_to_id)
            self.node_to_id[node] = node_id
            return node_id
        else:
            return self.node_to_id[node]
        
    def train_and_calculate_graphsage_embedding(self, init_dict, train_dict):
        graphsagemodel=GraphSAGE(**init_dict)
        
        start = time.process_time_ns()
        train_metrics = graphsagemodel.graphsage_train(**train_dict)
        end = time.process_time_ns()
        print(f"duration {(end-start)/1e9} sec")
        graphsagemodel.save_model(train_dict['dirs']+'/model_final')
        graphsagemodel.save_embedding(train_dict['dirs']+'/embedding_matrix_final')
        pickle.dump(self.node_to_id, open(train_dict['dirs']+'node_to_id.picle', 'wb'))
        self.embedding_to_pandas(graphsagemodel.get_embeddings()).to_parquet(self.config_path + 'embedding.parquet')
        
        return train_metrics 
    
    def print_metrics(self, train_metrics):
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
        
    def embedding_to_pandas(self, embedding):
        """calculates the node embedding and adds them in pandas df with original ID"""
        # embedding = np.load(self.model_path + '/embedding_matrix_final')
        df = pd.DataFrame(embedding)
        ids = dict((v,k) for k,v in self.node_to_id.items())
        id_df = pd.DataFrame.from_dict(ids, orient='index', columns=['id'])
        embed_df = id_df.join(df, how='outer')
        return embed_df
    
    def lin_grid_search(self, grid_dict, nodes, edges):
        grid_param = list(grid_dict.keys())[0]
        vals = grid_dict[grid_param]
        res = {}
        
        for val in vals:
            run = {}
            setattr(self, grid_param, val)
            train_metrics = self.get_embedding(nodes, edges)
            run['grid_param'] = grid_param
            run['grid_value'] = val
            run['loss'] = np.mean(train_metrics[-1]['train_loss'][-10:])
            run['val_loss'] = np.mean(train_metrics[-1]['val_loss'][-10:])
            run['train_metrics'] = train_metrics
            res[val]=run
                   
        pickle.dump(res, open(self.model_path + "grid_search_" + grid_param +".pickle" , "wb"))
        
        if self.verbose>=2:
            self.plot_grid(res)
            
        return res
    
    def plot_grid(self, res):
        losses = []
        val_losses = []
        for param_val, run in res.items():
            losses.append(run['loss'])
            val_losses.append(run['val_loss'])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        keys = [str(k) for k in res.keys()]
        ax1.bar(keys, losses, label='loss')
        ax1.bar(keys, val_losses, label='val_loss')
        for param_val, run in res.items():
            # ax2.plot(v['hist']['val_loss'], label=str(k))
            
            epoch_cnt = 0
            for episode_cnt, episode in enumerate(run['train_metrics']):
                epoch =  [x+epoch_cnt for x in episode['val_epoch']]
                ax2.plot(epoch, episode['val_loss'], label=str(param_val)+'_episode_' + str(episode_cnt))
                epoch_cnt = epoch_cnt + max(episode['epoch'])/ (len(episode['epoch']) - 1 ) * len (episode['epoch'])
            
        ax2.legend(bbox_to_anchor=(1.25, 1.0))
        print(f"loss: {losses}")
        print(f"val loss: {val_losses}")