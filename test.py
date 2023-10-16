#%% # create dummy feature matrixs
import numpy as np
import pandas as pd
import pickle

graphsage_embeddings_path = "graphsage_embeddings/bitcoin/embeddings.pkl"
node_embeddings_feature = pickle.load(open(graphsage_embeddings_path,"rb"))
print(f"dict size {len(node_embeddings_feature)}")
print(f"min key {min(node_embeddings_feature.keys())}")
print(f"max key {max(node_embeddings_feature.keys())}")
      

#%%
node_df = np.random.random(size=(3783, 11))
pdf = pd.DataFrame(node_df, columns = ["attr"+str(i) for i in range(11)])
pdf['id']=range(3783)
pdf.to_parquet("data/bitcoin/feature_attributes.parquet")



# %%
import os
import pickle
import networkx as nx 
os.chdir('../..')
import torch

cluster_id_hat = torch.load("data/enron/cluster_id_hat.pickle")
cluster_id = torch.load("data/enron/cluster_id.pickle")
# celoss_cluster = torch.nn.functional.cross_entropy(cluster_id_hat, cluster_id)
celoss_cluster = torch.nn.CrossEntropyLoss()

# %%
import pickle
import math
import torch.nn.functional as nnf
cluster_vector = pickle.load(open('temp/cluster_vector.pickle', 'rb'))
import numpy as np
import matplotlib.pyplot as plt
cnt_clusters = 8 + 2
cluster = [nnf.softmax(v, dim=-1) for v in cluster_vector]
cluster = np.concatenate([v.detach().numpy() for v in cluster], axis=0)
cluster = np.reshape(cluster, (1200, cnt_clusters))


fig = plt.figure(figsize=(20,20))
for i in range(cnt_clusters):
    ax = fig.add_subplot(math.ceil(cnt_clusters/2),2, i+1)
    ax.hist(cluster[:,i], bins=20)
# %%

true_cluster = pickle.load(open('temp/clusters.pickle', 'rb'))
true_cluster = np.concatenate([v.detach().numpy() for v in true_cluster], axis=0)
true_cluster = np.reshape(true_cluster, (1200,1))


pred_cluster = pickle.load(open('temp/cl_hat.pickle', 'rb'))
pred_cluster = np.concatenate([v.detach().numpy() for v in pred_cluster], axis=0)
pred_cluster = np.reshape(pred_cluster, (1200,1))

labels = ['pred', 'true']

for i, ds in enumerate([pred_cluster, true_cluster]):
    x_val, height = np.unique(ds, return_counts=True)
    plt.bar(x_val - 0.1 + 0.4*i, height=height, width=0.4, label=labels[i])
    

plt.legend()
plt.show()

# %% check synth walks 

obj = pickle.load(open('data/enron/synth_walks.pickle', 'rb'))
edges  = [(a,b) for (a,b,c) in obj]
edges_out = [a for (a,b) in edges]
synth_id, synth_cnt = np.unique(edges_out, return_counts=True)
log_synth_cnt = np.log10(synth_cnt)
plt.hist(log_synth_cnt)

# %%
import pandas as pd
edges = pd.read_parquet("data/enron/enron_edges.parquet")
cnt = edges['start'].value_counts()
log_cnt = np.log10(cnt)
plt.hist(log_cnt)
# %%
