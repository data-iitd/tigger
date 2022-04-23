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

from model_classes.transductive_model import EventLSTM

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
parser.add_argument("--l_w",default=17,help="lw", type=int)
parser.add_argument("--batch_size",default=20000,help="batch_size",type=int)




args = parser.parse_args()




data_path = args.data_path
gpu_num = args.gpu_num
config_dir = args.config_path
model_name = args.model_name
random_walk_sampling_rate  = args.random_walk_sampling_rate
num_of_sampled_graphs = args.num_of_sampled_graphs
l_w = args.l_w
sampling_batch_size = args.batch_size


#print("Input parameters")
print(args)
#print("########")


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
#print(data.head())
temporal_graph_original = defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: 0)))
for start,end,day in data[['start','end','days']].values:
    temporal_graph_original[day][start][end] += 1
    if undirected:
        temporal_graph_original[day][end][start] += 1
        
        
import h5py

vocab = pickle.load(open(config_dir+"/vocab.pkl","rb"))
time_stats = pickle.load(open(config_dir+"/time_stats.pkl","rb"))
mean_log_inter_time = time_stats['mean_log_inter_time']
std_log_inter_time = time_stats['std_log_inter_time']

pad_token = vocab['<PAD>']
print("Pad token", pad_token)

hf = h5py.File(config_dir+'/start_node_and_times.h5', 'r')
start_node_and_times = hf.get('1')
start_node_and_times = np.array(start_node_and_times)
start_node_and_times = list(start_node_and_times)
print("length of start node and times,", len(start_node_and_times))
hf.close()


if gpu_num == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(str(gpu_num)) if torch.cuda.is_available() else "cpu")

print("Computation device, ", device)

batch_size = 128
elstm = EventLSTM(vocab=vocab,nb_layers=2, nb_lstm_units=200,time_emb_dim= 64,
    embedding_dim=100, batch_size= batch_size,device=device,
    mean_log_inter_time=mean_log_inter_time,
    std_log_inter_time=std_log_inter_time)
elstm = elstm.to(device)
celoss = nn.CrossEntropyLoss(ignore_index=-1) #### -1 is padded     
optimizer = optim.Adam(elstm.parameters(), lr=.001)
num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)
print(" ##### Number of parameters#### " ,num_params)


best_model = torch.load(config_dir+"models/{}.pth".format(model_name),map_location=device)
elstm.load_state_dict(best_model)

isdir = os.path.isdir(config_dir+"/results") 
if not isdir:
    os.mkdir(config_dir+"/results")

for t in range(0, num_of_sampled_graphs):
    #print("Sampling iteration, ", t)
    import random
    #print("Random walk sampling rate", random_walk_sampling_rate)
    sampled_start_node_times = random.sample(start_node_and_times,data.shape[0]*random_walk_sampling_rate)
    #print("Required length of walks,", len(sampled_start_node_times))
    start_index = 0
    batch_size = sampling_batch_size
    generated_events= []
    generated_times = []
    #len(start_node_and_times)
    print("Length of required random walks", len(sampled_start_node_times))
    for start_index in range(0, len(sampled_start_node_times) ,batch_size):
        if start_index+batch_size < len(start_node_and_times):
            start_node = [[item[0]] for item in sampled_start_node_times[start_index:start_index+batch_size]]
            start_time = [[item[1]] for item in sampled_start_node_times[start_index:start_index+batch_size]]

            # Try with batch size of 10 or 15 since batch will be run in parallel
            # batch_X = [start_node]
            # batch_Xt = [start_time]
            batch_X = np.array(start_node)
            batch_Xt = np.array(start_time)
            pad_batch_X =  torch.LongTensor(batch_X).to(device)
            pad_batch_Xt =  torch.Tensor(batch_Xt).to(device)
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
                length += 1    
                X = pad_batch_X
                Xt = pad_batch_Xt
                batch_size, seq_len = X.size() 
                X= elstm.word_embedding(X)
                Xt = elstm.t2v(Xt)
                X = torch.cat((X, Xt), -1)
                X, elstm.hidden = elstm.lstm(X, elstm.hidden)
                X = X.contiguous()
                X = X.view(-1, X.shape[2])
                Y_hat = elstm.hidden_to_events(X)
                Y_hat = F.softmax(Y_hat,dim=-1)
                sampled_Y = torch.multinomial(Y_hat, 1, replacement=True)### (batch_size*seq_len)*number of replacements
                sampled_Y = sampled_Y + 1 ### Since event embedding starts from 1 , 0 is for padding
                Y= elstm.word_embedding(sampled_Y)
                Y = Y.view(-1, Y.shape[-1])
                X = torch.cat((X,Y),-1)
                X = elstm.sigmactivation(X)
                X = elstm.hidden_to_hidden_time(X)
                X = elstm.sigmactivation(X)
                X = X.view(batch_size,seq_len,X.shape[-1])  
                itd = elstm.lognormalmix.get_inter_time_dist(X) #### X is context 
                T_hat = itd.sample()
                T_hat = pad_batch_Xt.add(T_hat)
                T_hat = torch.round(T_hat)

                pad_batch_X = sampled_Y
                pad_batch_Xt = T_hat
                pad_batch_Xt[pad_batch_Xt < 1] = 1
                batch_generated_events.append(pad_batch_X.detach().cpu().numpy())
                batch_generated_times.append(pad_batch_Xt.detach().cpu().numpy())

            batch_generated_events = np.array(batch_generated_events).squeeze(-1).transpose()
            batch_generated_times = np.array(batch_generated_times).squeeze(-1).transpose()
            generated_events.append(batch_generated_events)
            generated_times.append(batch_generated_times)

    generated_events = np.concatenate(generated_events,axis=0)
    generated_times = np.concatenate(generated_times,axis=0)

    print(generated_events.shape,generated_times.shape)

    np.save(open(config_dir+"/results"+"/generated_events_{}.npy".format(str(t)),"wb"),generated_events)
    np.save(open(config_dir+"/results"+"/generated_times_{}.npy".format(str(t)),"wb"),generated_times)

reverse_vocab = {value:key for key, value in vocab.items()}
# end_node_id = vocab['end_node']
# ###### Clean the walks
# sampled_walks = []
# lengths =[]
# for i in range(generated_times.shape[0]):
#     sample_walk_event = []
#     sample_walk_time = []
#     done = False
#     j = 0
#     while not done and j <= l_w:
#         event = generated_events[i][j]
#         time = generated_times[i][j]
#         j += 1
#         if event == end_node_id or time > max_days:
#             done = True
#         else:
#             sample_walk_event.append(reverse_vocab[event])
#             sample_walk_time.append(time)
#     lengths.append(len(sample_walk_event))
#     sampled_walks.append((sample_walk_event,sample_walk_time))

# print("Mean length {} and Std deviation {}".format(str(np.mean(lengths)),str(np.std(lengths))))
os._exit(os.EX_OK)
