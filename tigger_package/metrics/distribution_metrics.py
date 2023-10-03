import math
import scipy 
import subprocess
import shlex
import os
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
from collections import defaultdict

class NodeDistributionMetrics:
    def __init__(self, nodes, synth_nodes, gtrie_dir=None, temp_dir=None):
        self.nodes = nodes
        self.synth_nodes = synth_nodes
        self.gtrie_dir = gtrie_dir
        self.temp_dir = temp_dir
        
    def calculate_wasserstein_distance(self):
        cols = self.nodes.columns
        
        ws_dist = {}
        
        for col in cols:
            ws = scipy.stats.wasserstein_distance(
                self.nodes[col].values,
                self.synth_nodes[col].values
            )
            ws_dist[col] = ws
            
        ws_df = pd.DataFrame.from_dict(ws_dist, orient='index', columns=['Wasserstein_dist'] )
        
        return ws_df
            
    def plot_hist(self):
        cols = self.nodes.columns
        
        cnt = len(cols)
        horz_plots = 4
        vert_plots = math.ceil(cnt/horz_plots)
        
        fig = plt.figure(figsize=(10,10))
        for i, col in enumerate(cols):
            ax = fig.add_subplot(vert_plots, horz_plots, i+1)
            ax.hist(self.nodes[col].values, bins=20, color='green', label='orig', alpha=0.5)
            ax.hist(self.synth_nodes[col].values, bins=20, color='red', label='synth', alpha=0.5)
            if i==cnt-1:
                ax.legend(bbox_to_anchor=(1.3, 1.))
                
class EdgeDistributionMetrics:
    def __init__(self, edges, synth_edges):
        self.edges = edges
        self.edges_degree = None
        self.synth_edges = synth_edges
        self.synth_edges_degree = None
        
        cols = list(self.edges.columns)
        cols.remove('end')
        cols.remove('start')
        self.cols = cols
        
        
    def calculate_wasserstein_distance(self):        
        ws_dist = {}
        
        for col in self.cols:
            ws = scipy.stats.wasserstein_distance(
                self.edges[col].values,
                self.synth_edges[col].values
            )
            ws_dist[col] = ws
            
        ws_df = pd.DataFrame.from_dict(ws_dist, orient='index', columns=['Wasserstein_dist'] )
        
        return ws_df
            
    def plot_hist(self):
        
        cnt = len(self.cols)
        horz_plots = 4
        vert_plots = math.ceil(cnt/horz_plots)
        
        fig = plt.figure(figsize=(10,10))
        for i, col in enumerate(self.cols):
            ax = fig.add_subplot(vert_plots, horz_plots, i+1)
            ax.hist(self.edges[col].values, bins=20, color='green', label='orig', alpha=0.5, density=True)
            ax.hist(self.synth_edges[col].values, bins=20, color='red', label='synth', alpha=0.5, density=True)
            if i==cnt-1:
                ax.legend(bbox_to_anchor=(1.3, 1.))
                
    def get_degrees_dist(self):
        if not self.edges_degree:
            for edge_name, name_dict in [('edges', {"src": 'start', 'dst': 'end'}), ('synth_edges', {"src": 'src', 'dst': 'dst'})]:
                edges = getattr(self, edge_name)
                out_degree = edges[name_dict['src']].value_counts(sort=False).value_counts(sort=False)
                in_degree = edges[name_dict['dst']].value_counts(sort=False).value_counts(sort=False)
                setattr(self, edge_name+"_degree", {'out_degree': out_degree, 'in_degree': in_degree})
                
    def get_degree_wasserstein_distance(self):
        self.get_degrees_dist()
        ws_dist = {}
        for direction in ['in_degree', 'out_degree']:
            ws = scipy.stats.wasserstein_distance(
                self.edges_degree[direction],
                self.synth_edges_degree[direction]
            )
            ws_dist[direction] = ws
            
        ws_df = pd.DataFrame.from_dict(ws_dist, orient='index', columns=['Wasserstein_dist'] )
        
        return ws_df
    
    def plot_degree_dst(self):
        self.get_degrees_dist()
        orig_in = self.edges_degree['in_degree']
        orig_out= self.edges_degree['out_degree']
        synth_in = self.synth_edges_degree['in_degree']
        synth_out = self.synth_edges_degree['out_degree']
        
        # calculate bins
        out_max = max(np.max(orig_out), np.max(synth_out))
        out_bins = [i* math.ceil(out_max/20) for i in range(20)]
        in_max =max(np.max(orig_in), np.max(synth_in))
        in_bins = [i* math.ceil(in_max/20) for i in range(20)]
        
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.hist(orig_in.index, weights=orig_in.values, bins=in_bins, alpha=0.5, label='orig', density=True, edgecolor='black')
        ax1.hist(synth_in.index, weights=synth_in.values, bins=in_bins, alpha=0.5, label='synth', density=True, edgecolor='black')
        ax2.hist(orig_out.index, weights=orig_out.values, bins=out_bins, alpha=0.5, label='orig', density=True, edgecolor='black')
        ax2.hist(synth_out.index, weights=synth_out.values, bins=out_bins, alpha=0.5, label='synth', density=True, edgecolor='black')
        
        ax1.legend()
        ax1.set_title("in degree dist")
        ax2.legend()
        ax2.set_title("out degree dist")
        fig.show()
        
    def widgets_distr(self):
        assert self.temp_dir is not None, "temp dir is not set"
        assert self.gtrie_dir is not None, "gtrie dir is not set"
        dfs = {}
        names = ['edges', 'synth_edges']
        
        for name in names:
            input_file = self.adj_to_csv(name)
            
            gtrie_cmd = f"{self.gtrie_dir}gtrieScanner -s 3 -m gtrie {self.gtrie_dir}gtries/dir3.gt -d -t html "
            input = f"-g {input_file} "
            output = f"-o {self.temp_dir}dir3_{name}.html "
   
            with open(self.temp_dir+"console_output_"+name, 'w') as fp:
                proc_res = subprocess.run(gtrie_cmd + input + output, shell=True, stdout=fp, stderr=fp)
                if proc_res.returncode < 0:
                    raise Exception("Terminal process did not exit succesfully")
            
            df = pd.read_html(f"{self.temp_dir}dir3_{name}.html")
            df =df[0][['Subgraph.1', 'Org. Frequency']]
            df[name+"_frac"] = df['Org. Frequency'] / df['Org. Frequency'].sum()
            df = df.rename({'Org. Frequency': name+"_freq"}, axis=1)
            dfs[name] = df
            
        df = dfs['edges'].merge(dfs['synth_edges'], on='Subgraph.1', how='outer')
        
        #calculate % difference
        df['delta'] = np.absolute(df['edges_frac'] - df['synth_edges_frac'])
        
        #plot results
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1, 1)
        
        x_axis = np.arange(df.shape[0])
                           
        ax.bar(x_axis - 0.2, df.edges_frac, width=0.4, label='orig', )
        ax.bar(x_axis + 0.2, df.synth_edges_frac, width=0.4, label='synth')
        ax.legend()
        ax.set_xticklabels(df['Subgraph.1'], rotation= '90')
        ax.set_xticks(np.arange(0, df.shape[0]))
        
        fig.show()

        return (df, df.delta.mean())
        
    def clustering_coef_undirected(self):
        """calculates the global clustering coef"""
        assert self.temp_dir is not None, "temp dir is not set"
        assert self.gtrie_dir is not None, "gtrie dir is not set"
        dfs = {}
        names = ['edges', 'synth_edges']
        
        for name in names:
            input_file = self.adj_to_csv(name)
   
            gtrie_cmd = f"{self.gtrie_dir}gtrieScanner -s 3 -m gtrie {self.gtrie_dir}gtries/undir3.gt -t html "
            input = f"-g {input_file} "
            output = f"-o {self.temp_dir}undir3_{name}.html "
   
            with open(self.temp_dir+"console_output_undir_"+name, 'w') as fp:
                proc_res = subprocess.run(gtrie_cmd + input + output, shell=True, stdout=fp, stderr=fp)
                if proc_res.returncode < 0:
                    raise Exception("Terminal process did not exit succesfully")
            
            df = pd.read_html(f"{self.temp_dir}dir3_{name}.html")
            df =df[0][['Subgraph.1', 'Org. Frequency']]
            df[name+"_frac"] = df['Org. Frequency'] / df['Org. Frequency'].sum()
            df = df.rename({'Org. Frequency': name+"_freq"}, axis=1)
            dfs[name] = df
            
        df = dfs['edges'].merge(dfs['synth_edges'], on='Subgraph.1', how='outer')
        traingles = df.loc[df['Subgraph.1']=='011 101 110']
        degree_dict = self.get_undirected_degrees()
        
        cc_orig = traingles['edges_freq'] / np.sum([k*(k-1) for k in degree_dict['edges']]) 
        
        return cc_orig
       
    def get_undirected_degrees(self):
        degree_dict = {}
        for edge_name, name_dict in [('edges', {"src": 'start', 'dst': 'end'}), ('synth_edges', {"src": 'src', 'dst': 'dst'})]:
            edges = getattr(self, edge_name)
            neighbors_dict = defaultdict(set)
            
            df = edges.loc[:,[name_dict['src'],name_dict['dst']]]
            for id, row in df.iterrows():
                start_id = row[name_dict['src']]
                end_id = row[name_dict['dst']]
                neighbors_dict[start_id].add(end_id)
                neighbors_dict[end_id].add(start_id)
                            
            degrees = [len(v) for v in neighbors_dict.values()]
            degree_dict[edge_name] = degrees
        return degree_dict
            
                    
    def adj_to_csv(self, name):
        input_file = self.temp_dir + name + "_adj.csv"
        df = getattr(self, name)
        if name == 'edges':
            df = df.rename({'start': 'src', 'end': 'dst'}, axis=1)
            
        df['new_src'] = df['src'] + 1
        df['new_dst'] = df['dst'] + 1
        
        df[['new_src', 'new_dst']].to_csv(input_file, header=False, index=False, sep=" ", compression=None)
        return input_file
        