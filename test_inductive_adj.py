import os
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
import csv
from torch.autograd import Variable
from torch.nn import functional as F

from model_classes.inductive_model import EventClusterLSTM

from metrics.metric_utils import get_numpy_matrix_from_adjacency,get_adj_graph_from_random_walks,get_total_nodes_and_edges_from_temporal_adj_list_in_time_range,get_adj_origina_graph_from_original_temporal_graph
from metrics.metric_utils import sample_adj_graph_multinomial_k_inductive,sample_adj_graph_topk
from metrics.metrics import compute_graph_statistics,calculate_mmds,calculate_mmd_distance,create_timestamp_edges,calculate_temporal_katz_index,Edge
import sklearn

from tgg_utils import *

### configurations 
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="full path of original dataset in csv format(start,end,time)",
                    type=str)
parser.add_argument("--gpu_num",help="GPU no. to use, -1 in case of no gpu", type=int)
parser.add_argument("--config_path",help="full path of the folder where models and related data are saved during training", type=str)
parser.add_argument("--model_name",help="name of the model need to be loaded", type=str)
parser.add_argument("--random_walk_sampling_rate", help="No. of epochs to be sampled from random walks",type=int)
parser.add_argument("--num_of_sampled_graphs",help="No. of times , a graph to be sampled", type=int)
parser.add_argument("--graph_sage_embedding_path",help="GraphSage embedding path",type=str)
parser.add_argument("--l_w",default=15,help="lw", type=int)
parser.add_argument("--gan_embedding_path",default="",help="GAN embedding path",type=str)


args = parser.parse_args()
print(args)
data_path = args.data_path
gpu_num = args.gpu_num
config_dir = args.config_path
model_name = args.model_name
random_walk_sampling_rate  = args.random_walk_sampling_rate
num_of_sampled_graphs = args.num_of_sampled_graphs
graph_sage_embedding_path = args.graph_sage_embedding_path
l_w = args.l_w
gan_embedding_path=args.gan_embedding_path

print("Input parameters")
print(args)
print("########")


strictly_increasing_walks = True
num_next_edges_to_be_stored = 100
undirected = True

data= pd.read_csv(data_path)
data = data[['start','end','days']]
node_set = set(data['start']).union(set(data['end']))
print("number of nodes,",len(node_set))
node_set.update('end_node')
max_days = max(data['days'])
print("Minimum, maximum timestamps",min(data['days']),max_days)
data = data.sort_values(by='days',inplace=False)
print("number of interactions," ,data.shape[0])
print(data.head())



temporal_graph_original = defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: 0)))
for start,end,day in data[['start','end','days']].values:
    temporal_graph_original[day][start][end] += 1
    if undirected:
        temporal_graph_original[day][end][start] += 1
        
        
import h5py
node_embeddings_feature = pickle.load(open(graph_sage_embedding_path,"rb"))
vocab = pickle.load(open(config_dir+"/vocab.pkl","rb"))
cluster_labels = pickle.load(open(config_dir+"/cluster_labels.pkl","rb"))
pca = pickle.load(open(config_dir+"/pca.pkl","rb"))
kmeans = pickle.load(open(config_dir+"/kmeans.pkl","rb"))
time_stats = pickle.load(open(config_dir+"/time_stats.pkl","rb"))
mean_log_inter_time = time_stats['mean_log_inter_time']
std_log_inter_time = time_stats['std_log_inter_time']
pad_cluster_id = cluster_labels[0]
print("Pad cluster id, ", pad_cluster_id)

num_components = np.max(cluster_labels) + 1
print(num_components)
pad_token = vocab['<PAD>']
print("Pad token", pad_token)
node_embedding_matrix = np.load(open(config_dir+"/node_embedding_matrix.npy","rb"))
if gan_embedding_path!="":
    print("Gan embedding provided")
    generated_embeddings = np.load(open(gan_embedding_path,"rb"))
    node_embedding_matrix = np.concatenate((node_embedding_matrix[:2],generated_embeddings),axis=0)
print("Final node embedding matrix,",node_embedding_matrix.shape)
normalized_dataset = node_embedding_matrix[1:] / np.linalg.norm(node_embedding_matrix[1:], axis=1)[:, np.newaxis]
import h5py
hf = h5py.File(config_dir+'start_node_and_times.h5', 'r')
start_node_and_times = hf.get('1')
start_node_and_times = np.array(start_node_and_times)
start_node_and_times = list(start_node_and_times)
hf.close()

import scann

searcher = scann.scann_ops_pybind.builder(normalized_dataset, 20, "dot_product").tree(
    num_leaves=200, num_leaves_to_search=1000, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()


if gpu_num == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(str(gpu_num)) if torch.cuda.is_available() else "cpu")

print("Computation device, ", device)

batch_size = 512  ### can be changed
elstm = EventClusterLSTM(vocab=vocab,node_pretrained_embedding = node_embedding_matrix,nb_layers=2, nb_lstm_units=128,time_emb_dim= 64,
        embedding_dim=128, batch_size= batch_size,device=device,
        mean_log_inter_time=mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,num_components=num_components)
elstm = elstm.to(device)
celoss = nn.CrossEntropyLoss(ignore_index=-1) #### -1 is padded     
optimizer = optim.Adam(elstm.parameters(), lr=.001)
num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)
print(" ##### Number of parameters#### " ,num_params)


best_model = torch.load(config_dir+"models/{}.pth".format(model_name),map_location=device)['model']
elstm.load_state_dict(best_model)
elstm.eval()
isdir = os.path.isdir(config_dir+"/results") 
if not isdir:
    os.mkdir(config_dir+"/results")

for t in range(0, num_of_sampled_graphs):
    print("Sampling iteration, ", t)
    import random
    print("Random walk sampling rate", random_walk_sampling_rate)
    sampled_start_node_times = random.sample(start_node_and_times,data.shape[0]*random_walk_sampling_rate)
    print("Selected,", len(sampled_start_node_times))
    start_index = 0
    print("Change it to 5000 in case of other large datasets")
    batch_size = 1024 
    generated_events= []
    generated_times = []
    #len(start_node_and_times)
    print("Length of required random walks", len(sampled_start_node_times))
    num_epochs = len(sampled_start_node_times)/batch_size
    for start_index in range(0, len(sampled_start_node_times) ,batch_size):
        print("\r%d/%d" %(int(start_index/batch_size),num_epochs),end="")
        if start_index+batch_size < len(sampled_start_node_times):
            start_node = [[item[0]] for item in sampled_start_node_times[start_index:start_index+batch_size]]
            start_time = [[item[1]] for item in sampled_start_node_times[start_index:start_index+batch_size]]
            start_cluster = [[item[3]] for item in sampled_start_node_times[start_index:start_index+batch_size]]
            batch_X = np.array(start_node)
            batch_Xt = np.array(start_time)
            batch_Cx = np.array(start_cluster)

            pad_batch_X =  torch.LongTensor(batch_X).to(device)
            pad_batch_Xt =  torch.Tensor(batch_Xt).to(device)
            pad_batch_Cx = torch.LongTensor(batch_Cx).to(device)

            #print(pad_batch_X.shape,pad_batch_Xt.shape)

            hidden_a = torch.zeros(elstm.nb_lstm_layers, pad_batch_X.shape[0], elstm.nb_lstm_units).to(device)
            hidden_b = torch.zeros(elstm.nb_lstm_layers, pad_batch_X.shape[0], elstm.nb_lstm_units).to(device)
            elstm.hidden = (hidden_a,hidden_b)
            length = 0
            batch_generated_events= []
            batch_generated_times = []
            batch_generated_events.append(pad_batch_X.detach().cpu().numpy())
            batch_generated_times.append(pad_batch_Xt.detach().cpu().numpy())
            while length < l_w:
                #print("\r%d"%length,end='')
                #print(length)
                length += 1    
                X = pad_batch_X
                Xt = pad_batch_Xt
                XCID = pad_batch_Cx
                batch_size, seq_len = X.size() 
                X= elstm.word_embedding(X)
                Xt = elstm.t2v(Xt)
                XCID_embedding = elstm.cluster_embeddings(XCID)
                X = torch.cat((X, Xt,XCID_embedding), -1)
                X, elstm.hidden = elstm.lstm(X, elstm.hidden)
                X = X.contiguous()
                X = X.view(-1, X.shape[2])
                #print(X.shape)
                # run through actual Event linear layer
                Y_hat = elstm.hidden_to_ne_hidden(X)
                Y_hat = elstm.relu1(Y_hat) ### Introducing non-linearity
                Y_hat = Y_hat.view(batch_size, seq_len,Y_hat.shape[-1])
                Y_clusterid = elstm.clusterid_hidden(Y_hat)
                Y_clusterid = F.softmax(Y_clusterid,dim=-1)
                Y_clusterid = Y_clusterid.view(-1,Y_clusterid.shape[-1])
                Y_clusterid = torch.multinomial(Y_clusterid, 1, replacement=True)
                Y_clusterid = Y_clusterid.view(batch_size,seq_len)
                pad_batch_Cx = Y_clusterid.clone()
                Y_clusterid = Y_clusterid.unsqueeze(-1).repeat(1,1,elstm.mu_hidden_dim).unsqueeze(2)

                mu, log_var = elstm.ne_mu(Y_hat), elstm.ne_var(Y_hat)
                #print(mu.shape,log_var.shape)
                mu = mu.view(batch_size,seq_len,elstm.num_components,elstm.mu_hidden_dim)
                #print(mu.shape)
                log_var = log_var.view(batch_size,seq_len,elstm.num_components,elstm.mu_hidden_dim)
                #print(log_var.shape)
                mu = torch.gather(mu,2,Y_clusterid).squeeze(2)
                log_var = torch.gather(log_var,2,Y_clusterid).squeeze(2)
                std = torch.exp(log_var / 2)
                q = torch.distributions.Normal(mu, std)
                z = q.rsample()
                ne_hat = elstm.ne_decoder1(elstm.relu2(elstm.ne_decoder(z)))
                decoder_mu = elstm.decoder_mu(ne_hat)
                decoder_std = elstm.decoder_std(ne_hat)
                ne_hat = decoder_mu
                X = torch.cat((X,ne_hat.view(-1,ne_hat.shape[2])),-1)
                #print(X.shape)
                X = elstm.sigmactivation(X)
                X = elstm.hidden_to_hidden_time(X)
                X = elstm.sigmactivation(X)
                #print(X.shape)
                # Run through actual Time Linear layer

                X = X.view(batch_size,seq_len,X.shape[-1])  
            ##print("Here ,",X.shape)
                itd = elstm.lognormalmix.get_inter_time_dist(X) #### X is context 
                T_hat = itd.sample()
                T_hat = pad_batch_Xt.add(T_hat)
                T_hat = torch.round(T_hat)
                ne_hat = ne_hat.view(ne_hat.shape[0],-1)
                sampled_Y, distances = searcher.search_batched(ne_hat.detach().cpu().numpy(), leaves_to_search=100, pre_reorder_num_neighbors=1000)
                sampled_Y = sampled_Y[:,:1]
                sampled_Y = torch.LongTensor(np.int32(sampled_Y+1)).to(device)
                pad_batch_X = sampled_Y
                pad_batch_Xt = T_hat
                pad_batch_Xt[pad_batch_Xt < 1] = 1
                #print(pad_batch_X.shape,pad_batch_Xt.shape)
                batch_generated_events.append(pad_batch_X.detach().cpu().numpy())
                batch_generated_times.append(pad_batch_Xt.detach().cpu().numpy())
            #print("\n")
            batch_generated_events = np.array(batch_generated_events).squeeze(-1).transpose()
            batch_generated_times = np.array(batch_generated_times).squeeze(-1).transpose()
            generated_events.append(batch_generated_events)
            generated_times.append(batch_generated_times)
            #print("Length of ",len(generated_events))
    generated_events = np.concatenate(generated_events,axis=0)
    generated_times = np.concatenate(generated_times,axis=0)


    print("\n",generated_events.shape,generated_times.shape)

    np.save(open(config_dir+"/results"+"/generated_events_{}.npy".format(str(t)),"wb"),generated_events)
    np.save(open(config_dir+"/results"+"/generated_times_{}.npy".format(str(t)),"wb"),generated_times)

#     reverse_vocab = {value:key for key, value in vocab.items()}
#     end_node_id = vocab['end_node']
#     ###### Clean the walks
#     sampled_walks = []
#     lengths =[]
#     for i in range(generated_times.shape[0]):
#         sample_walk_event = []
#         sample_walk_time = []
#         done = False
#         j = 0
#         while not done and j <= l_w:
#             event = generated_events[i][j]
#             time = generated_times[i][j]
#             j += 1
#             if event == end_node_id or time > max_days:
#                 done = True
#             else:
#                 sample_walk_event.append(reverse_vocab[event])
#                 sample_walk_time.append(time)
#         lengths.append(len(sample_walk_event))
#         sampled_walks.append((sample_walk_event,sample_walk_time))
#     print("Mean length {} and Std deviation {}".format(str(np.mean(lengths)),str(np.std(lengths))))