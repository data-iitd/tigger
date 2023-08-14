#%%
import os
import pickle
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy as np
import copy
import torch
import torch.optim as optim
from tgg_utils import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
# import scann
print("loaded")
from model_classes.inductive_model import get_topk_event_prediction_rate
from model_classes.edge_node_lstm import EdgeNodeLSTM

#%%
feature_path = "data/bitcoin/feature_attributes.parquet"
data_path = "data/bitcoin/edgelist_with_attributes.parquet"
gpu_num = -1
config_path = "temp/"
num_epochs = 10
graphsage_embeddings_path = "graphsage_embeddings/bitcoin/embeddings.pkl"
num_clusters = 500
window_interactions = 6
l_w = 20
filter_walk = 2
data= pd.read_parquet(data_path)


#%% Prep node_set
node_set = set(data['start']).union(set(data['end']))
print("number of nodes,",len(node_set))
# node_set.update('end_node')
print("number of interactions," ,data.shape[0])
edge_attr_cols = [c for c in data.columns if c not in ['start', 'end']]
print(f"attributes found for edges: {edge_attr_cols}")

#%% create node and edge objects with links lists.
edges = []
node_id_to_object = {}
for row in data.values:
    start = row[0] 
    end = row[1]
    
    # add start and end node to node_dict
    if start not in node_id_to_object:
        node_id_to_object[start] = Node(id=start, as_start_node=[], as_end_node=[])
    if end not in node_id_to_object:
        node_id_to_object[end] = Node(id=end, as_start_node=[], as_end_node=[])
        
    #add edge to edge dict
    edge= Edge(start=start,end=end, attributes=row[2:], outgoing_edges = [],incoming_edges=[])
    edges.append(edge) 
    node_id_to_object[start].as_start_node.append(edge)  # add edge to start node list
    node_id_to_object[end].as_end_node.append(edge)      # add edge to end node list.
print("length of edges,", len(edges), " length of nodes,", len(node_id_to_object))


#%% define the sample + prob per edge
ct = 0
#for edge in tqdm(edges): 
for edge in edges:
    end_node_edges = node_id_to_object[edge.end].as_start_node  #get succesive edges
    edge.outgoing_edges = end_node_edges
    edge.out_nbr_sample_probs = prepare_sample_probs(edge)
    ct += 1

dist = [len(e.outgoing_edges) for e in edges]
plt.hist(dist)
plt.xscale("log")


#%% create a vocab for the LSTM
graph_original = defaultdict(lambda: defaultdict(lambda: 0))
for start,end in data[['start','end']].values:
    graph_original[start][end] += 1
vocab = {'<PAD>': 0,'end_node':1}
for node in data['start']:
    if node not in vocab:
        vocab[node] = len(vocab)
for node in data['end']:
    if node not in vocab:
        vocab[node] = len(vocab)
print("Length of vocab, ", len(vocab))
print("Id of end node , ",vocab['end_node'])
pad_token = vocab['<PAD>']


#%% create random walks + sequences
def sample_random_Walks():
    print("Running Random Walk, Length of edges, ", len(edges))
    random_walks = []
    for edge in edges:
        random_walks.append(run_uniform_random_walk(edge,l_w))
    print("length of collected random walks,", len(random_walks))
    random_walks = [item for item in random_walks if item is not None]
    print("length of collected random walks after removing None,", len(random_walks))
    # random_walks = [clean_random_walk(item) for item in random_walks]  # not needed
    random_walks = [item for item in random_walks if filter_rw(item,filter_walk)]  # ensure minimum walk length
    print("Length of random walks after removing short ranadom walks", len(random_walks))
    return random_walks

def get_sequences_from_random_walk(random_walks, vocab):
    sequences = [convert_walk_to_edge_node_seq(walk, vocab) for walk in random_walks]
    return sequences

random_walks = sample_random_Walks()
sequences = get_sequences_from_random_walk(random_walks, vocab)
print("Average length of random walks")
lengths = []
for wk in random_walks:
    lengths.append(len(wk))
print("Mean length {} and Std deviation {}".format(str(np.mean(lengths)),str(np.std(lengths))))

#%% create node embedding matrix

node_embeddings_feature = pickle.load(open(graphsage_embeddings_path,"rb"))  ### We tried deep walk or node2vec embeddings

node_emb_size = 128
node_embedding_matrix = np.zeros((len(vocab),node_emb_size))
for item in vocab:
    if item == '<PAD>':
        arr = np.zeros(node_emb_size)
    elif item == 'end_node':
        arr = np.random.uniform(-0.1,0.1,node_emb_size)
    else:
        arr = node_embeddings_feature[item]
    index= vocab[item]
    node_embedding_matrix[index] = arr
print("Node embedding matrix, shape,", node_embedding_matrix.shape)
# create row normalized dataset excluding <padding>
normalized_dataset = node_embedding_matrix / np.linalg.norm(node_embedding_matrix, axis=1)[:, np.newaxis] 
normalized_dataset[0] = 0  #set padding back to 0 
#%%

# searcher = scann.scann_ops_pybind.builder(normalized_dataset, 20, "dot_product").tree(
#     num_leaves=200, num_leaves_to_search=1000, training_sample_size=250000).score_ah(
#     2, anisotropic_quantization_threshold=0.2).reorder(100).build()
# searcher_1 = scann.scann_ops_pybind.builder(normalized_dataset, 1, "dot_product").tree(
#     num_leaves=1000, num_leaves_to_search=1000, training_sample_size=3000).score_ah(
#     2, anisotropic_quantization_threshold=0.2).reorder(100).build()

searcher = KDTree(node_embedding_matrix[1:], leaf_size=40)
searcher_1 = KDTree(normalized_dataset[1:], leaf_size=40)

#%% reduce dim and cluster
pca = PCA(n_components=3)  # PCA(n_components=110)
node_embedding_matrix_pca = pca.fit_transform(node_embedding_matrix[2:]) ### since 0th and 1st index is of padding and end node
print("PCA variance explained",np.sum(pca.explained_variance_ratio_))

kmeans = KMeans(n_clusters=num_clusters, random_state=0,max_iter=10000).fit(node_embedding_matrix_pca)
labels = kmeans.labels_
label_freq = Counter(labels)
print("max size cluster, ", max(label_freq.values()))
cluster_labels = [0,1]+[item+2 for item in labels]
#print(cluster_labels)
print(len(cluster_labels))
max_label = np.max(cluster_labels)
print("Max cluster label",max_label)

pad_cluster_id = cluster_labels[0]
print("Pad cluster id, ", pad_cluster_id)

num_components = np.max(cluster_labels) + 1
print("num_components ",num_components)

reverse_vocab= {value:key for key,value in vocab.items() }

def add_cluster_id(sequence,labels):
    # edge attribute, node id in vocab, cluster_id
    return [(a,b,labels[b]) for a,b in sequence]
sequences = [add_cluster_id(sequence,cluster_labels) for sequence in sequences]

#%% prepaire shuffled and padded batches
config_dir = config_path ### Change in random walks
isdir = os.path.isdir(config_dir) 
if not isdir:
    os.mkdir(config_dir)
isdir = os.path.isdir(config_dir+"/models") 
if not isdir:
    os.mkdir(config_dir+"/models")

def get_X_Y_T_CID_from_sequences(sequences):  ### This also need to provide the cluster id of the 
    seq_X = []
    seq_Y = []
    seq_Xedge = []
    seq_Yedge = []
    seq_XCID = []
    seq_YCID = []
    for seq in sequences:
        seq_Xedge.append([list(item[0]) for item in seq[:-1]])  ## O contains edge attributes
        seq_Yedge.append([list(item[0]) for item in seq[1:]])
        seq_X.append([item[1] for item in seq[:-1]])   ## 1 contain vocab[node_id]
        seq_Y.append([item[1] for item in seq[1:]])
        seq_XCID.append([item[2] for item in seq[:-1]])   ## 2 contains the cluster id
        seq_YCID.append([item[2] for item in seq[1:]])
    X_lengths = [len(sentence) for sentence in seq_X]
    Y_lengths = [len(sentence) for sentence in seq_Y]
    max_len = max(X_lengths)
    return seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, max_len, seq_XCID, seq_YCID

def get_batch(start_index,batch_size, seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID, max_len):
    """Creates padded batch copied to the torch device"""
    edge_attr_dim = len(seq_Xedge[0][0])  # dimension of the edge attributes
    
    batch_X = seq_X[start_index:start_index+batch_size]
    batch_Y = seq_Y[start_index:start_index+batch_size]
    batch_Xedge = seq_Xedge[start_index:start_index+batch_size]
    batch_Yedge = seq_Yedge[start_index:start_index+batch_size]    
    batch_X_len = X_lengths[start_index:start_index+batch_size]
    batch_Y_len = Y_lengths[start_index:start_index+batch_size]
    batch_XCID = seq_XCID[start_index:start_index+batch_size]
    batch_YCID = seq_YCID[start_index:start_index+batch_size] 
    
    pad_batch_X = np.ones((batch_size, max_len),dtype=np.int64)*pad_token
    pad_batch_Y = np.ones((batch_size, max_len),dtype=np.int64)*pad_token
    pad_batch_Xedge = np.ones((batch_size, max_len, edge_attr_dim),dtype=np.float32)*pad_token
    pad_batch_Yedge = np.ones((batch_size, max_len, edge_attr_dim),dtype=np.float32)*pad_token
    pad_batch_XCID = np.ones((batch_size, max_len),dtype=np.int64)*pad_cluster_id
    pad_batch_YCID = np.ones((batch_size, max_len),dtype=np.int64)*pad_cluster_id
    
    for i, x_len in enumerate(batch_X_len):
        #print(i,x_len,len(batch_X[i][:x_len]),len(pad_batch_X[i, 0:x_len]))
        pad_batch_X[i, 0:x_len] = batch_X[i][:x_len]
        pad_batch_Y[i, 0:x_len] = batch_Y[i][:x_len]
        pad_batch_Xedge[i, 0:x_len] = batch_Xedge[i][:x_len]
        pad_batch_Yedge[i, 0:x_len] = batch_Yedge[i][:x_len]
        pad_batch_XCID[i, 0:x_len] = batch_XCID[i][:x_len]
        pad_batch_YCID[i, 0:x_len] = batch_YCID[i][:x_len]
        
    pad_batch_X =  torch.LongTensor(pad_batch_X).to(device)
    pad_batch_Y =  torch.LongTensor(pad_batch_Y).to(device)
    pad_batch_Xedge =  torch.Tensor(pad_batch_Xedge).to(device)
    pad_batch_Yedge =  torch.Tensor(pad_batch_Yedge).to(device)
    batch_X_len = torch.LongTensor(batch_X_len).to(device)
    batch_Y_len = torch.LongTensor(batch_Y_len).to(device)
    pad_batch_XCID =  torch.LongTensor(pad_batch_XCID).to(device)
    pad_batch_YCID =  torch.LongTensor(pad_batch_YCID).to(device)
    
    return pad_batch_Xedge, pad_batch_Yedge, pad_batch_X, pad_batch_Y, batch_X_len, batch_Y_len, pad_batch_XCID, pad_batch_YCID

def data_shuffle(seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID):
    indices = list(range(0, len(seq_X)))
    random.shuffle(indices)
    #### Data Shuffling
    seq_X = [seq_X[i] for i in indices]   #### 
    seq_Y = [seq_Y[i] for i in indices]
    seq_Xedge = [seq_Xedge[i] for i in indices]
    seq_Yedge = [seq_Yedge[i] for i in indices]    
    X_lengths = [X_lengths[i] for i in indices]
    Y_lengths = [Y_lengths[i] for i in indices]
    seq_XCID = [seq_XCID[i] for i in indices]
    seq_YCID = [seq_YCID[i] for i in indices]
    return seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID

#TODO
def evaluate_model(elstm,batch_size=128):
    elstm.eval()
    running_loss= []
    running_event_loss = []
    running_time_loss= []
    event_prediction_rates = []
    topk_event_prediction_rates = []
    topk10_event_prediction_rates = []
    topk20_event_prediction_rates = []

    random_walks = sample_random_Walks()
    sequences = get_sequences_from_random_walk(random_walks, vocab)
    sequences = [add_cluster_id(sequence,cluster_labels) for sequence in sequences]
    
    seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, max_len, seq_XCID, seq_YCID = get_X_Y_T_CID_from_sequences(sequences)
    seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID = data_shuffle(seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID)
    
    for start_index in range(0, len(seq_X),batch_size):

        if start_index+batch_size < len(seq_X) and start_index + batch_size < 10000:
            #try:
                pad_batch_Xedge, pad_batch_Yedge, pad_batch_X, pad_batch_Y, batch_X_len, batch_Y_len, pad_batch_XCID, pad_batch_YCID \
                    = get_batch(start_index,batch_size, seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID, max_len)
                mask_distribution = pad_batch_Y!=0
                mask_distribution = mask_distribution.to(device)
                num_events_time_loss = mask_distribution.sum().item()

                Y_hat, edge_hat, loss, log_dict, Y_clusterid = elstm(
                    X=pad_batch_X, Y=pad_batch_Y,
                    Xedge=pad_batch_Xedge, Yedge = pad_batch_Yedge,
                    XCID=pad_batch_XCID, YCID=pad_batch_YCID,
                    X_lengths= batch_X_len, mask=mask_distribution, kl_weight=kl_weight
                    )
                Y = pad_batch_Y
                Y = Y.view(-1)
                Y_hat = Y_hat.view(-1, Y_hat.shape[-1])   
                mask_distribution = mask_distribution.view(-1)
                Y_hat = Y_hat.detach().cpu().numpy()
                Y = Y.detach().cpu().numpy()
                # neighbors, distances = searcher.search_batched(node_embedding_matrix[:100], leaves_to_search=1000, pre_reorder_num_neighbors=1000)
                Y_hat = searcher.query(Y_hat, 20, return_distance=False)
                Y_hat = Y_hat+1  # correct for the padding value at index 0 that is not in searcher
                topk_event_prediction_rates.append(get_topk_event_prediction_rate(Y,Y_hat, k=5,ignore_Y_value=0))
                topk10_event_prediction_rates.append(get_topk_event_prediction_rate(Y,Y_hat, k=10,ignore_Y_value=0))
                topk20_event_prediction_rates.append(get_topk_event_prediction_rate(Y,Y_hat, k=20,ignore_Y_value=0))

    print("Event prediction rate@top5:,", np.mean(topk_event_prediction_rates))
    print("Event prediction rate@top10:,", np.mean(topk10_event_prediction_rates))
    print("Event prediction rate@top20:,", np.mean(topk20_event_prediction_rates))

#%% start of training

if gpu_num == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(str(gpu_num)) if torch.cuda.is_available() else "cpu")

print("Computation device, ", device)

batch_size = 1024  
wt_update_ct = 0
debug = False
print_ct = 100
best_loss = 1000000000
best_model = None 
epoch_wise_loss = []
kl_weight = 0.00001


elstm = EdgeNodeLSTM(
    vocab=vocab, 
    node_pretrained_embedding = normalized_dataset,
    nb_layers=2, 
    nb_lstm_units=128,
    edge_emb_dim= sequences[0][0][0].shape[0],
    clust_dim=64, # used for cluster embedding
    batch_size= batch_size,
    device=device,
    num_components=num_components)  # number of clusters
elstm = elstm.to(device)
optimizer = optim.Adam(elstm.parameters(), lr=.001)
num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)
print(" ##### Number of parameters#### " ,num_params)

#%%  

for epoch in range(0,num_epochs+1):
    print("KL weight is , ", kl_weight)
    elstm.train()
    try:
        print(f"Epoch: {epoch}")
    except:
        print("Seqx not defined")
    running_loss= []
    running_event_loss = []
    running_time_loss= []
    running_recon_loss = []
    running_kl_loss = []
    running_cluster_loss = []
    
    event_prediction_rates = []
    topk_event_prediction_rates = []
    topk10_event_prediction_rates = []

    random_walks = sample_random_Walks()
    sequences = get_sequences_from_random_walk(random_walks, vocab)
    sequences = [add_cluster_id(sequence,cluster_labels) for sequence in sequences]
    
    seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, max_len, seq_XCID, seq_YCID = get_X_Y_T_CID_from_sequences(sequences)
    seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID = data_shuffle(seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID)
    
    for start_index in range(0,1): #range(0, len(seq_X),batch_size):
        if start_index+batch_size < len(seq_X):
            print("\r%d/%d" %(int(start_index),len(seq_X)),end="")
            pad_batch_Xedge, pad_batch_Yedge, pad_batch_X, pad_batch_Y, batch_X_len, batch_Y_len, pad_batch_XCID, pad_batch_YCID \
                = get_batch(start_index,batch_size, seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID, max_len)
            elstm.zero_grad()
            mask_distribution = pad_batch_Y!=0
            mask_distribution = mask_distribution.to(device)
            num_events_time_loss = mask_distribution.sum().item()
            
            # forward + backward pas
            Y_hat, edge_hat, loss, log_dict, Y_clusterid = elstm(
                X=pad_batch_X, Y=pad_batch_Y,
                Xedge=pad_batch_Xedge, Yedge = pad_batch_Yedge,
                XCID=pad_batch_XCID, YCID=pad_batch_YCID,
                X_lengths= batch_X_len, mask=mask_distribution, kl_weight=kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(elstm.parameters(), 1)
            optimizer.step()
            
            
            running_loss.append(loss.item())
            running_event_loss.append(log_dict['elbo']), 
            running_cluster_loss.append(log_dict['cross_entropy_cluster'])
            running_recon_loss.append(log_dict['reconstruction_ne'])
            running_kl_loss.append(log_dict['kl'])
            wt_update_ct += 1
            if wt_update_ct%print_ct == 0 and debug:
                print("Running Loss :, ",np.mean(running_loss[-print_ct:]) )
                print("Running event elbo loss: ", np.mean(running_event_loss[-print_ct:]),np.std(running_event_loss[-print_ct:]))
                print("Running cluster ce loss: ", np.mean(running_cluster_loss[-print_ct:]),np.std(running_cluster_loss[-print_ct:]))
                print("Running recon elbo loss: ", np.mean(running_recon_loss[-print_ct:]),np.std(running_recon_loss[-print_ct:]))
                print("Running kl elbo loss: ", np.mean(running_kl_loss[-print_ct:]),np.std(running_kl_loss[-print_ct:]))
                print()

    print("\nEpoch done")
    print_ct = len(running_loss)
    print("Running Loss :, ",np.mean(running_loss[-print_ct:]) )
    print("Running event elbo loss: ", np.mean(running_event_loss[-print_ct:]),np.std(running_event_loss[-print_ct:]))
    print("Running time log loss: ", np.mean(running_time_loss[-print_ct:]),np.std(running_time_loss[-print_ct:]))
    print("Running cluster ce loss: ", np.mean(running_cluster_loss[-print_ct:]),np.std(running_cluster_loss[-print_ct:]))
    print("Running recon elbo loss: ", np.mean(running_recon_loss[-print_ct:]),np.std(running_recon_loss[-print_ct:]))
    print("Running kl elbo loss: ", np.mean(running_kl_loss[-print_ct:]),np.std(running_kl_loss[-print_ct:]))
    #break
    if epoch%20 == 0:
        print("Running evaluation")
        evaluate_model(elstm)
    state = {
        'model':elstm.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss':np.mean(running_loss)
    }
    epoch_wise_loss.append(np.mean(running_loss))
    # if epoch%20 == 0:
    #     torch.save(state, config_dir+"/models/{}.pth".format(str(epoch)))
    #     import h5py
    #     hf = h5py.File(config_dir+'/start_node_and_times.h5', 'w')
    #     hf.create_dataset('1', data=start_node_and_times)
    #     hf.close()
    if np.mean(running_loss) < best_loss:
        print("### Saving the best model ####")
        best_loss = np.mean(running_loss)
        best_elstm = copy.deepcopy(elstm.state_dict())
        torch.save(state, config_dir+"/models/best_model.pth".format(str(epoch)))


# %%
