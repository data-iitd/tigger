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
from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
from stellargraph import StellarGraph
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE

from stellargraph import datasets
from stellargraph.utils import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model
#%%
dataset = datasets.Cora()
# display(HTML(dataset.description))
G, node_subjects = dataset.load()

fullbatch_generator = FullBatchNodeGenerator(G, sparse=False)
gcn_model = GCN(layer_sizes=[128], activations=["relu"], generator=fullbatch_generator)

corrupted_generator = CorruptedGenerator(fullbatch_generator)
gen = corrupted_generator.flow(G.nodes())
# %%
infomax = DeepGraphInfomax(gcn_model, corrupted_generator)
x_in, x_out = infomax.in_out_tensors()

model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
epochs = 100
es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
plot_history(history)
# %%
x_emb_in, x_emb_out = gcn_model.in_out_tensors()

# for full batch models, squeeze out the batch dim (which is 1)
x_out = tf.squeeze(x_emb_out, axis=0)
emb_model = Model(inputs=x_emb_in, outputs=x_out)
test_gen = fullbatch_generator.flow(G.nodes())


test_embeddings = emb_model.predict(test_gen)
# %%
from tensorflow import keras
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

nodes = list(G.nodes())
number_of_walks = 1
length = 5

unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks
)

batch_size = 50
epochs = 4
num_samples = [10, 5]

generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
train_gen = generator.flow(unsupervised_samples)

layer_sizes = [50, 50]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
)

# Build the model and expose input and output sockets of graphsage, for node pair inputs:
x_inp, x_out = graphsage.in_out_tensors()

# %%
prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)
model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)
# %%
history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)
# %%

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
# %%
node_ids = node_subjects.index
node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)


# %%
