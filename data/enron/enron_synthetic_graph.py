#%%
# this notebook creates a synthetic graph based on the enron graph.

#%%
import os
import pickle
import networkx as nx
# os.chdir('../..')
os.getcwd()
from tigger_package.orchestrator import Orchestrator
from tigger_package.tools import plot_adj_matrix
enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
#%% load graph and convert from nx to node and adjency dataframes
import tigger_package

# import importlib
# importlib.reload(tigger_package)
# from tigger_package.tools import nx_to_df

# enron_folder = "data/enron/"
# enron_graph_file = "enron_sub_graph4.pickle"
# node_file = "enron_nodes.parquet"
# edge_file = "enron_edges.parquet"

# graph = pickle.load(open(enron_folder + enron_graph_file, "rb"))
# nodes, edges = nx_to_df(graph)
# nodes = nodes.drop(['label', 'old_id'], axis=1)
# edges.rename({'source': "start", 'target': "end"}, inplace=True, axis=1)
# nodes.to_parquet(enron_folder + node_file)
# edges.to_parquet(enron_folder + edge_file)

# %% gridsearch graphsage

# grid_res = orchestrator.lin_grid_search_graphsage({"learning_rate": [0.001, 0.0001, 0.00001,0.000001]})
# best model is 0.00001
# epoch is based on overfitting on val_loss curve 

# %% Final Model
train_metrics = orchestrator.create_embedding()

#%% flownet grid

# res = orchestrator.lin_grid_search_flownet({"learning_rate": [0.001, 0.0001, 0.00001]})
# 0.0001 with 250 epoch
# res = orchestrator.lin_grid_search_flownet({"number_of_bijectors": [4,6,10]})
# 4 with 500 epoch loss -8.98, val losss -8,91


#%% flownet final model

# name, history = orchestrator.train_flow()
# orchestrator.sample_flownet()
# %% gridsearch lstm

# enron_folder = "data/enron/"
# orchestrator = Orchestrator(enron_folder)
# res = orchestrator.lin_grid_search_lstm({'dropout': [0.3]})
# %% train lstm and create synthetic walks
# enron_folder = "data/enron/"
# orchestrator = Orchestrator(enron_folder)
# loss_dict = orchestrator.train_lstm()
# orchestrator.create_synthetic_walks(target_cnt=200, synth_node_file_name='sampled_flownet')
# orchestrator.generate_synth_graph()


#%% full run
import tigger_package.metrics.distribution_metrics
import tigger_package.orchestrator
enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)

# train_metrics = orchestrator.create_embedding()

# name, history = orchestrator.train_flow()
# orchestrator.sample_flownet()

loss_dict = orchestrator.train_lstm()
# orchestrator.create_synthetic_walks(target_cnt=2000, synth_node_file_name='sampled_flownet')
# orchestrator.generate_synth_graph()


# %%
import pandas as pd
adj_df = pd.read_parquet("data/enron/synth_graph/adjacency.parquet")
plot_adj_matrix(adj_df)
# %% BASE

import tigger_package.metrics.distribution_metrics
import tigger_package.orchestrator

import importlib
importlib.reload(tigger_package.metrics.distribution_metrics)
importlib.reload(tigger_package.orchestrator)
from tigger_package.metrics.distribution_metrics import NodeDistributionMetrics, EdgeDistributionMetrics
from tigger_package.orchestrator import Orchestrator

enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
# nodes = orchestrator._load_nodes()
# synth_nodes = orchestrator._load_synthetic_graph_nodes()
# ndm = NodeDistributionMetrics(nodes, synth_nodes)
# ndm.calculate_wasserstein_distance()
# ndm.plot_hist()

#%%% CLUSTERING

# edges = orchestrator._load_edges()
# synth_edges = orchestrator._load_synth_graph_edges()
import pandas as pd
edges = pd.DataFrame([[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]], columns=['start', 'end'])
synth_edges = pd.DataFrame([[1,2], [1,3], [1,4], [2,3],[3,4] ], columns=['src', 'dst'])
edm = EdgeDistributionMetrics(edges, synth_edges)
# edm.calculate_wasserstein_distance()
# edm.plot_hist()
# edm.get_degree_wasserstein_distance()
# edm.plot_degree_dst()


edm.gtrie_dir = '~/Downloads/gtrieScanner_src_01/'
edm.temp_dir = 'temp/'
# df, mean = edm.widgets_distr()
cc = edm.clustering_coef_undirected()
print(cc)
# %%
import networkx as nx
import numpy as np
enron_folder = "data/enron/"
enron_graph_file = "enron_sub_graph4.pickle"
graph = pickle.load(open(enron_folder + enron_graph_file, "rb"))
G2 = graph.to_undirected()
# nx.transitivity(graph.to_undirected())
np.sum(list(nx.triangles(G2).values()))
# %%

edm.edges
# (3*3971) / 118880