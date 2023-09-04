import pickle
import os
import pandas as pd
import numpy as np
import yaml

import tigger_package.graphsage.graphsage_controller
import tigger_package.flownet
import tigger_package.inductive_controller

import importlib
importlib.reload(tigger_package.graphsage.graphsage_controller)
importlib.reload(tigger_package.flownet)
importlib.reload(tigger_package.inductive_controller)
from tigger_package.flownet import FlowNet
from tigger_package.inductive_controller import InductiveController
from tigger_package.graphsage.graphsage_controller import GraphSageController

class Orchestrator():
    def __init__(self, config_path):
        with open(config_path + "config.yaml", 'r') as file:
            config_dict = yaml.safe_load(file)
        self.config = config_dict
        self.config_path = config_path
        self.flownet = None
        self.gsc = None
        self.inductiveController = None
    
    def train_flow(self):
        node = self._load_nodes()
        embed = self.load_normalized_embed()
        self.flownet = FlowNet(
            config_path=self.config_path,
            config_dict=self.config['flownet'])
        name, hist = self.flownet.train(embed, node)
        return (name, hist)
    
    def sample_flownet(self, name=None):
        self.flownet.sample_model(self.config['target_node_count'], name)
        
    def lin_grid_search_flownet(self, grid_dict):
        node = self._load_nodes()
        embed = self.load_normalized_embed()
        if not self.flownet:
            self.flownet = FlowNet(
                config_path=self.config_path,
                config_dict=self.config['flownet'])
        res = self.flownet.lin_grid_search(grid_dict, embed, node)
        return res
    
    def create_embedding(self):
        nodes = self._load_nodes()
        edges =  self._load_edges()
        self.gsc = GraphSageController(
            path=self.config_path,
            config_dict=self.config['graphsage']
        )
        train_metrics = self.gsc.get_embedding(nodes, edges)
        return train_metrics
        
    def lin_grid_search_graphsage(self, grid_dict):
        nodes = self._load_nodes()
        edges =  self._load_edges()
        if not self.flownet:
            self.gsc = GraphSageController(
                path=self.config_path,
                config_dict=self.config['graphsage']
            )
        res = self.gsc.lin_grid_search(grid_dict, nodes, edges)
        return res
          
    def train_lstm(self):
        nodes = self._load_nodes()
        edges =  self._load_edges()
        embed = self._load_embed()
        self.inductiveController = InductiveController(
            nodes=nodes,
            edges=edges,
            embed=embed,
            path=self.config_path,
            config_dict=self.config['lstm']
        )
        epoch_wise_loss, loss_dict = self.inductiveController.train_model()
        return (epoch_wise_loss, loss_dict)
       
    def lin_grid_search_lstm(self, grid_dict):
        nodes = self._load_nodes()
        edges =  self._load_edges()
        embed = self._load_embed()
        if not self.inductiveController:
            self.inductiveController = InductiveController(
                nodes=nodes,
                edges=edges,
                embed=embed,
                path=self.config_path,
                config_dict=self.config['lstm']
            )
        res = self.inductiveController.lin_grid_search(grid_dict)
        return res 
                                                       
    # -- private methodes
    
    def _load_edges(self):
        return pd.read_parquet(self.config_path + self.config['edges_path'])  

    def _load_nodes(self):
        # assume id column
        nodes = pd.read_parquet(self.config_path + self.config['nodes_path'])  
        nodes = nodes.sort_values('id').set_index('id')
        return nodes
    
    def load_normalized_embed(self):
        embed_path = self.config_path + self.config['embed_path']
        try:  # incase embed is stored as dict
            node_embeddings = pickle.load(open(embed_path,"rb"))                   
            node_embedding_df = pd.DataFrame.from_dict(node_embeddings, orient='index')
        except:
            node_embedding_df = pd.read_parquet(embed_path)
            node_embedding_df = node_embedding_df.sort_values('id').set_index('id')
        
        norm = np.linalg.norm(node_embedding_df, ord=np.inf, axis=1)
        node_embedding_df.div(norm, axis=0)
        node_embedding_df.fillna(1)
        
        return node_embedding_df
    
    def _load_embed(self):
        embed_path = self.config_path + self.config['embed_path']
        try:  # incase embed is stored as dict
            node_embeddings = pickle.load(open(embed_path,"rb"))                   
            node_embedding_df = pd.DataFrame.from_dict(node_embeddings, orient='index')
        except:
            node_embedding_df = pd.read_parquet(embed_path)
            node_embedding_df = node_embedding_df.sort_values('id').set_index('id')
        
        return node_embedding_df
        
        
    
            
    
    
    
        