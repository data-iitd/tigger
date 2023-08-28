import pickle
import os
import pandas as pd
import numpy as np
import yaml

import tigger_package.flownet

import importlib
importlib.reload(tigger_package.flownet)
from tigger_package.flownet import FlowNet

class Orchestrator():
    def __init__(self, config_path):
        with open(config_path + "config.yaml", 'r') as file:
            config_dict = yaml.safe_load(file)
        self.config = config_dict
        self.config_path = config_path
        self.flownet = None

    def load_nodes(self):
        return pd.read_parquet(self.config_path + self.config['nodes_path'])  
    
    def load_normalized_embed(self):
        embed_path = self.config_path + self.config['embed_path']
        try:  # incase embed is stored as dict
            node_embeddings = pickle.load(open(embed_path,"rb"))                   
            node_embedding_df = pd.DataFrame.from_dict(node_embeddings, orient='index')
        except:
            raise NotImplementedError("only loading embed from pickled dict is implemented")
        
        norm = np.linalg.norm(node_embedding_df, ord=np.inf, axis=1)
        node_embedding_df.div(norm, axis=0)
        node_embedding_df.fillna(1)
        
        return node_embedding_df
    
    def train_flow(self):
        node = self.load_nodes()
        embed = self.load_normalized_embed()
        self.flownet = FlowNet(
            config_path=self.config_path,
            config_dict=self.config['flownet'])
        name, hist = self.flownet.train(embed, node)
        return (name, hist)
    
    def sample_flownet(self, name=None):
        self.flownet.sample_model(self.config['target_node_count'], name)
        
    def lin_grid_search_flownet(self, grid_dict):
        node = self.load_nodes()
        embed = self.load_normalized_embed()
        if not self.flownet
            self.flownet = FlowNet(
                config_path=self.config_path,
                config_dict=self.config['flownet'])
        self.flownet.lin_grid_search(self, grid_dict, embed, node)
        
        
    
            
    
    
    
        