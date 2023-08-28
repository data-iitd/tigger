import os
import random
import pickle
import pandas as pd
import numpy as np

class GraphGenerator:
    def __init__(self, results_dir, node_cols, edge_cols, seed=1):
        self.results_dir = results_dir  
        self.node_cols = node_cols  
        self.edge_cols = edge_cols
        
        random.seed(seed)
        os.makedirs(results_dir, exist_ok=True)
    
    def generate_node_df(self, nodes):
        # remove end node
        if np.mean(nodes.iloc[-1])==1:
            end_node_id = nodes.shape[0]-1
            nodes = nodes.drop(index=end_node_id)
        node_dim = len(self.node_cols)
        
        node_attr = nodes.iloc[:, -node_dim:]
        node_attr.columns = self.node_cols
        node_attr.to_parquet(self.results_dir + "/node_attributes.parquet")
        return node_attr
        
    def to_adjacency_df(self, sampled_edges):
        adjacency_list = []
        for edge in sampled_edges:
            adjacency_list.append([edge[0]] + [edge[1]] + edge[2])
        
        adjacency_df = pd.DataFrame(adjacency_list)
        adjacency_df.columns = ['src', 'dst'] + self.edge_cols
        adjacency_df.to_parquet(self.results_dir + "/adjacency.parquet")        
        return adjacency_df
    
    def sample_edges(self, edges, target_edge_count):
        """Samples target_edge_count edge from edges avoiding duplicate edges"""
        processed = set()
        sampled = []
        
        while len(sampled)<target_edge_count and len(edges)>0:
            edge = edges.pop(random.randrange(len(edges)))
            key = str(edge[0]) + "_" + str(edge[1])
            # edge is not already sampled and not a self edge
            if key not in processed and edge[0]!=edge[1]:
                sampled.append(edge)
                processed.add(key)
                
        return sampled
        
    def generate_graph(self, nodes, edges, target_edge_count):
        gen_nodes = self.generate_node_df(nodes)
        sampled_edges = self.sample_edges(edges, target_edge_count)
        adjacency_df = self.to_adjacency_df(sampled_edges)
        return (gen_nodes, adjacency_df)
    
    def load_edges_and_nodes(self, node_path, edge_path):
        nodes = pd.read_parquet(node_path)
        edges = pickle.load(open(edge_path, "rb"))
        return nodes, edges
        