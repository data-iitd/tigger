#%%
import pandas as pd
import numpy as np
import pickle
import os 
os.chdir('../..')
os.getcwd()
#%% edgelist
output_path = 'data/test_graph/'
edge_attr_cnt = 3
node_attr_cnt = 5
node_cnt = 11
emb_dim = 16

#%% edgelist
edge_dict = {'start': range(node_cnt-1), 'end': range(1,node_cnt)}

# for i in range(edge_attr_cnt):
#     if i%2==0:
#         edge_dict['attr'+str(i)] = range(node_cnt-1)
#     else:
#         edge_dict['attr'+str(i)] = range(1, node_cnt)
edge_list = pd.DataFrame(edge_dict)
#swap last edge
edge_list.iloc[9, 0] = 8


for i in range(edge_attr_cnt):
    if i%2==0:
        edge_list['attr'+str(i)] = edge_list['start']
    else:
        edge_list['attr'+str(i)] = edge_list['end']

edge_list.to_parquet(output_path + "test_edge_list.parquet")

edge_list
# %%
node_dict = {'id': range(node_cnt)}
for i in range(node_attr_cnt):
    node_dict['attr'+str(i)] = range(node_cnt)

node_df = pd.DataFrame(node_dict)
node_df.to_parquet(output_path + 'test_node_attr.parquet')
# %%
embedding_dict = {}
for n in range(node_cnt):
      embedding_dict[n] = np.ones(16, dtype=float) * n * 2 / node_cnt
      
with open(output_path + 'test_embedding.pickle', 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
start_node_dict = {}
for i in range(16):
    start_node_dict['attr'+str(i)] = [0.1]*30
for i in range(16, 19):
    start_node_dict['attr'+str(i)] = [0.2]*30

node_df = pd.DataFrame(start_node_dict)
node_df
node_df.to_parquet(output_path + 'synth_nodes.parquet')

# %%

import pandas as pd
import numpy as np
import pickle
import os 
os.chdir('../..')
os.getcwd()
#%%
import tigger_package.inductive_controller
# import tigger_package.edge_node_lstm
import importlib
importlib.reload(tigger_package.inductive_controller)
# importlib.reload(tigger_package.edge_node_lstm)
from tigger_package.inductive_controller import InductiveController


if __name__ == "__main__":
    output_path = 'data/test_graph/'
    node_feature_path = output_path + 'test_node_attr.parquet'
    edge_list_path = output_path + "test_edge_list.parquet"
    graphsage_embeddings_path = output_path + 'test_embedding.pickle'
    n_walks=7
    inductiveController = InductiveController(
        node_feature_path=node_feature_path,
        edge_list_path=edge_list_path,
        graphsage_embeddings_path=graphsage_embeddings_path,
        n_walks=n_walks,
        batch_size = 6,
        num_clusters = 5,
        l_w = 6
    )
    seqs = inductiveController.sample_random_Walks()
    seqs = inductiveController.get_X_Y_from_sequences(seqs)
    seqs = inductiveController.data_shuffle(seqs)
    seqs = inductiveController.get_batch(0, 6, seqs)
    
    epoch_wise_loss, loss_dict = inductiveController.train_model()

# %%


generated_nodes = pd.read_parquet("data/test_graph/synth_nodes.parquet")
print(f"there are {generated_nodes.shape[0]} nodes generated")
gen_edges = inductiveController. create_synthetic_walks(generated_nodes, 2)
pickle.dump(gen_edges, open("data/test_graph/generated_edges.pickle", "wb"))
print(f"there are {generated_nodes.shape[0]} nodes after edge generation")

#%%
import tigger_package.graph_generator
importlib.reload(tigger_package.graph_generator)
from tigger_package.graph_generator import GraphGenerator

graph_generator = GraphGenerator(
    results_dir = 'data/test_graph/results' , 
    node_cols = inductiveController.node_features.columns, 
    edge_cols = inductiveController.edge_attr_cols
)

generated_nodes, gen_edges = graph_generator.load_edges_and_nodes(
    node_path = "data/test_graph/synth_nodes.parquet",
    edge_path = "data/test_graph/generated_edges.pickle"
)
res = graph_generator.generate_graph(
    nodes=generated_nodes,
    edges=gen_edges,
    target_edge_count=len(inductiveController.data)
)

#%%
import tigger_package.orchestrator

import importlib
importlib.reload(tigger_package.orchestrator)
from tigger_package.orchestrator import Orchestrator

import tigger_package.flownet
importlib.reload(tigger_package.flownet)
from tigger_package.flownet import FlowNet

# embed_path = 'data/test_graph/test_embedding.pickle'
# nodes_path =  'data/test_graph/test_node_attr.parquet'
config_path = 'data/test_graph/'
orchestrator = Orchestrator(config_path)

node = orchestrator._load_nodes()
embed = orchestrator.load_normalized_embed()
x_data =  embed.join(node, how='inner')
x_data = x_data.drop('id', axis =1)
# name, history = orchestrator.train_flow()
# orchestrator.sample_flownet()
res = orchestrator.lin_grid_search_flownet({"learning_rate": [0.001, 0.0001]})
print("hoi")
# %%
import matplotlib.pyplot as plt
losses = []
val_losses = []
for k, v in res.items():
    losses.append(v['loss'])
    val_losses.append(v['val_loss'])

fig, (ax1, ax2) = plt.subplots(1, 2)
keys = [str(k) for k in res.keys()]
ax1.bar(keys, losses, label='loss')
ax1.bar(keys, val_losses, label='val_loss')
for k, v in res.items():
    ax2.plot(v['hist']['val_loss'], label=str(k))
ax2.legend()
fig.show()
# %%
