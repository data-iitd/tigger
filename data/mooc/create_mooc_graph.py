#%%
import os
import pandas as pd
import numpy as np
print(os.getcwd())

#%%
SOURCE_PATH = 'original_data/'  #input graph
MOOC_ACTION_FILE = 'mooc_actions.tsv'
MOOC_FEATURES_FILE = 'mooc_action_features.tsv'
MOOC_LABEL_FILE = 'mooc_action_labels.tsv'
FEATURE_NAMES = ['FEATURE0', 'FEATURE1', 'FEATURE2', 'FEATURE3', 'weight']

#%%
# def prep_edge_data():
mooc_action = pd.read_csv(SOURCE_PATH + MOOC_ACTION_FILE, sep='\t')
features = pd.read_csv(SOURCE_PATH + MOOC_FEATURES_FILE, sep='\t')
edges = mooc_action.merge(features, on='ACTIONID', how='inner')
print(f"edges: {edges.shape[0]} mooc_action {mooc_action.shape[0]}")

# load and merge labels. Labels contain duplicates / and have missing
labels = pd.read_csv(SOURCE_PATH + MOOC_LABEL_FILE, sep='\t')
labels =labels.drop_duplicates(subset='ACTIONID')
edges = edges.merge(labels, on='ACTIONID', how='left')
edges = edges.fillna(0)

edges['src'] = "U" + edges['USERID'].astype(str)
edges['dst'] = "T" + edges['TARGETID'].astype(str)
edges

#%% CREATE NODE DATA
user_nodes = edges.groupby('src').agg({'TARGETID': 'size', 'LABEL': 'sum'})
user_nodes['is_user'] = 1
user_nodes.rename(columns={'TARGETID': 'degree'}, inplace=True)

course_nodes = edges.groupby('dst').agg({'USERID': 'size', 'LABEL': 'sum'})
course_nodes['is_user'] = 0
course_nodes.rename(columns={'USERID': 'degree'}, inplace=True)
nodes = pd.concat([user_nodes, course_nodes])

# normalize min max normalize features
for fld in ['degree', 'LABEL', 'is_user']:
    col = nodes[fld]
    nodes[fld] = (col-col.min())/(col.max()-col.min())

nodes = nodes.reset_index()
nodes = nodes.rename(columns={'index': 'old_id'})
nodes = nodes.reset_index()
nodes = nodes.rename(columns={'index': 'id'})
# nodes[['id', 'degree', 'LABEL', 'is_user']].to_parquet("mooc_nodes.parquet")

#%% CREATE EDGES

#update node id
edges = (edges
         .merge(nodes[['old_id', 'id']], left_on='src', right_on='old_id', how='inner')
         .drop(['src', 'old_id'], axis=1)
         .rename(columns={'id': 'start'})
)

edges = (edges
         .merge(nodes[['old_id', 'id']], left_on='dst', right_on='old_id', how='inner')
         .drop(['dst', 'old_id'], axis=1)
         .rename(columns={'id': 'end'})
)

#above edges from a multi graph. Therefore we group by src ,dst
# and take avg of features + cnt column
edges = (edges.groupby(['start', 'end'])
            .agg({'FEATURE0': 'mean', 'FEATURE1': 'mean', 'FEATURE2': 'mean', 'FEATURE3': 'mean', 'TIMESTAMP': 'size'})
            .rename(columns={'TIMESTAMP': 'weight'})
            .reset_index()
)

# normalize min max normalize features
for fld in FEATURE_NAMES:
    col = edges[fld]
    edges[fld] = (col-col.min())/(col.max()-col.min())

edges.to_parquet("mooc_edges.parquet")



#%%


extended_edges = edges.merge(labels, on='ACTIONID', how='inner')
    print(f"input_df {train_edges.shape[0]}, after adding labels {graph_edges.shape[0]}")

    # remove ACTIONID filed
    graph_edges = graph_edges.drop("ACTIONID", axis='columns')
    print(f"edge colunms {graph_edges.columns}")
    G = nx.from_pandas_edgelist(graph_edges, 'src', 'dst', edge_attr=True, create_using=nx.DiGraph)
    print(f"edge check: df:{graph_edges.shape[0]}, graph edges: {G.number_of_edges()}")
    print(f"node check: users+ courses:{graph_edges['src'].nunique() + graph_edges['dst'].nunique()}, nodes: {G.number_of_nodes()}")

    #update node attribute
    for n in G.nodes(data=True):
        if n[0].startswith("U"):
            n[1]['attr_is_user']=1
        else:
            n[1]['attr_is_user']=0

    G = nx.convert_node_labels_to_integers(G, label_attribute='label')
    return G