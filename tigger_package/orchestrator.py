import pickle
import os
import pandas as pd
import numpy as np
import yaml
import tensorflow as tf

import tigger_package.graphsage.graphsage_controller
import tigger_package.flownet
import tigger_package.inductive_controller
import tigger_package.graph_generator

import importlib
importlib.reload(tigger_package.graphsage.graphsage_controller)
importlib.reload(tigger_package.flownet)
importlib.reload(tigger_package.inductive_controller)
importlib.reload(tigger_package.graph_generator)

from tigger_package.graph_generator import GraphGenerator
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
        with tf.device('/CPU:0'):
            node = self._load_nodes()
            embed = self.load_normalized_embed()
            self.flownet = FlowNet(
                config_path=self.config_path,
                config_dict=self.config['flownet'])
            name, hist = self.flownet.train(embed, node)
        return (name, hist)
    
    def sample_flownet(self, model_name=None):
        name = self.config_path + self.config['synth_nodes']
        if model_name:
            self.flownet = FlowNet(
                config_path=self.config_path,
                config_dict=self.config['flownet'])
            self.flownet.load_model(model_name)
        self.flownet.sample_model(self.config['target_node_count'], name)
        
    def lin_grid_search_flownet(self, grid_dict):
        with tf.device('/CPU:0'):
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
        if not self.inductiveController:
            self.init_lstm()
        loss_dict = self.inductiveController.train_model()
        return (loss_dict)
    
    def init_lstm(self):
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
       
    def lin_grid_search_lstm(self, grid_dict):
        if not self.inductiveController:
            self.init_lstm()
        res = self.inductiveController.lin_grid_search(grid_dict)
        return res 
    
    def create_synthetic_walks(self, target_cnt, synth_node_file_name=None, map_real_time=True):
        generated_nodes = self._load_synthetic_nodes(synth_node_file_name)
        self.synth_walks = self.inductiveController.create_synthetic_walks(generated_nodes, target_cnt=target_cnt, map_real_time=map_real_time)
        pickle.dump(self.synth_walks, open(self.config_path + self.config['synth_walks'], "wb"))
     
    def generate_synth_graph(self):
        results_dir = self.config_path + self.config['synth_graph_dir']
        if not self.inductiveController:
            self.init_lstm()            
        
        graph_generator = GraphGenerator(
            results_dir = results_dir , 
            node_cols = self.inductiveController.node_features.columns, 
            edge_cols = self.inductiveController.edge_attr_cols
        )
        
        graph_generator.generate_graph(
            nodes=self._load_synthetic_nodes(),
            edges=self._load_synth_walks(),
            target_edge_count=len(self.inductiveController.data)
        )
           
                                                       
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
        
    def _load_synthetic_nodes(self, name=None):
        """loads the synth node embed_ + attrib from flownet"""
        path = self.config_path + self.config['synth_nodes']
        
        synth_nodes = pd.read_parquet(path)
        return synth_nodes
    
    def _load_synth_walks(self):
        path = self.config_path + self.config['synth_walks']
        synth_walk = pickle.load(open(path, ("rb"))) 
        return synth_walk   
                
    
    def _load_synthetic_graph_nodes(self, name=None):
        path = self.config_path + self.config['synth_graph_dir'] + 'node_attributes.parquet'
        return pd.read_parquet(path)
    
    def _load_synth_graph_edges(self):
        path = self.config_path + self.config['synth_graph_dir'] + 'adjacency.parquet'  
        return pd.read_parquet(path)
    
        