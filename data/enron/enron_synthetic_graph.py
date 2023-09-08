#%%
# this notebook creates a synthetic graph based on the enron graph.

#%%
import os
import pickle
import networkx as nx 
os.chdir('../..')
os.getcwd()

#%% load graph and convert from nx to node and adjency dataframes
# import tigger_package

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

# %%
import importlib
import tigger_package.orchestrator
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator

enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
grid_res = orchestrator.lin_grid_search_graphsage({"learning_rate": [0.001, 0.0001, 0.00001,0.000001]})
# best model is 0.00001
# epoch is based on overfitting on val_loss curve 

# %% Final Model
import importlib
import tigger_package.orchestrator
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator
enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
train_metrics = orchestrator.create_embedding()

#%% flownet grid
import importlib
import tigger_package.orchestrator
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator

enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
# res = orchestrator.lin_grid_search_flownet({"learning_rate": [0.001, 0.0001, 0.00001]})
# 0.0001 with 250 epoch
res = orchestrator.lin_grid_search_flownet({"number_of_bijectors": [4,6,10]})
# 4 with 500 epoch loss -8.98, val losss -8,91


#%% flownet final model
import importlib
import tigger_package.orchestrator
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator

enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
name, history = orchestrator.train_flow()
orchestrator.sample_flownet(name='sampled_flownet')
# %% gridsearch lstm

import importlib
import tigger_package.orchestrator
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator

enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
res = orchestrator.lin_grid_search_lstm({'weight_decay': [0.01, 0.001, 0.0001]})
# %% train lstm

import importlib
import tigger_package.orchestrator
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator

enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)
loss_dict = orchestrator.train_lstm()
