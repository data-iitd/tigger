import random
import pandas as pd
from datetime import datetime
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from multiprocessing import Pool

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
from tgg_utils import *
import argparse
from model_classes.transductive_model import EventLSTM,get_event_prediction_rate,get_time_mse,get_topk_event_prediction_rate


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="full path of dataset in csv format(start,end,time)",
                    type=str)
parser.add_argument("--gpu_num",help="GPU no. to use, -1 in case of no gpu", type=int)
parser.add_argument("--config_path",help="full path of the folder where models and related data need to be saved", type=str)
parser.add_argument("--num_epochs",default=200,help="Number of epochs for training", type=int)
parser.add_argument("--window_interactions",default=6,help="Interaction window", type=int)
parser.add_argument("--l_w",default=20,help="lw", type=int)
parser.add_argument("--filter_walk",default=2,help="filter_walk", type=int)




args = parser.parse_args()
print(args)
data_path = args.data_path
gpu_num = args.gpu_num
config_path = args.config_path
num_epochs = args.num_epochs
window_interactions = args.window_interactions
l_w = args.l_w
filter_walk = args.filter_walk
data= pd.read_csv(data_path)

data = data[['start','end','days']]
node_set = set(data['start']).union(set(data['end']))
print("number of nodes,",len(node_set))
node_set.update('end_node')
max_days = max(data['days'])
#data = data.drop_duplicates(['start','end','days'])
print("Minimum, maximum timestamps",min(data['days']),max_days)
data = data.sort_values(by='days',inplace=False)
print("number of interactions," ,data.shape[0])

#print(data.head())

### configurations 

strictly_increasing_walks = True
num_next_edges_to_be_stored = 100

edges = []
node_id_to_object = {}
undirected=True
for start,end,days in data[['start','end','days']].values:
    if start not in node_id_to_object:
        node_id_to_object[start] = Node(id=start,as_start_node= [],as_end_node=[])
    if end not in node_id_to_object:
        node_id_to_object[end] = Node(id=end,as_start_node= [],as_end_node=[])
        
    edge= Edge(start=start,end=end,time=days,outgoing_edges = [],incoming_edges=[])
    edges.append(edge) 
    node_id_to_object[start].as_start_node.append(edge)
    node_id_to_object[end].as_end_node.append(edge)
    if undirected:
        edge= Edge(start=end,end=start,time=days,outgoing_edges = [],incoming_edges=[])
        edges.append(edge) 
        node_id_to_object[end].as_start_node.append(edge)
        node_id_to_object[start].as_end_node.append(edge)        
print("length of edges,", len(edges), " length of nodes,", len(node_id_to_object))

ct = 0
#for edge in tqdm(edges): 
for edge in edges:
        
    end_node_edges = node_id_to_object[edge.end].as_start_node
    index = binary_search_find_time_greater_equal(end_node_edges,edge.time,strictly=strictly_increasing_walks)
    if index != -1:
        if strictly_increasing_walks:
            edge.outgoing_edges = end_node_edges[index:index+num_next_edges_to_be_stored]
        else:
            edge.outgoing_edges = [item for item in end_node_edges[index:index+num_next_edges_to_be_stored] if item.end != edge.start]

    start_node_edges = node_id_to_object[edge.start].as_end_node
    index = binary_search_find_time_lesser_equal(start_node_edges,edge.time,strictly=strictly_increasing_walks)
    if index != -1:
        if strictly_increasing_walks:
            edge.incoming_edges = start_node_edges[max(0,index-num_next_edges_to_be_stored):index+1]
        else:
            edge.incoming_edges = [item for item in start_node_edges[max(0,index-num_next_edges_to_be_stored):index+1] if item.start != edge.end]
        edge.incoming_edges.reverse()

    ct += 1
# for edge in tqdm(edges):
#     edge.outgoing_edges= sort_edges_timewise(edge.outgoing_edges,reverse=False)
#     edge.incoming_edges= sort_edges_timewise(edge.incoming_edges,reverse=True)
#### Learn Neighbour sampling method ####
for edge in tqdm(edges):
    edge.out_nbr_sample_probs = []
    edge.in_nbr_sample_probs = []

    if len(edge.outgoing_edges) >= 1:
        edge.out_nbr_sample_probs,edge.outJ, edge.outq =  prepare_alias_table_for_edge(edge,incoming=False,window_interactions=window_interactions) ### Gaussian Time Sampling
        
    if len(edge.incoming_edges) >= 1:
        edge.in_nbr_sample_probs,edge.inJ, edge.inq =  prepare_alias_table_for_edge(edge,incoming=True,window_interactions=2)

temporal_graph_original = defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: 0)))
for start,end,day in data[['start','end','days']].values:
    temporal_graph_original[day][start][end] += 1
    if undirected:
        temporal_graph_original[day][end][start] += 1
vocab = {'<PAD>': 0,'end_node':1}
for node in data['start']:
    if node not in vocab:
        vocab[node] = len(vocab)
for node in data['end']:
    if node not in vocab:
        vocab[node] = len(vocab)
#print("Length of vocab, ", len(vocab))
#print("Id of end node , ",vocab['end_node'])
pad_token = vocab['<PAD>']
def run_parallel_walk(edge):
    return run_random_walk_without_temporal_constraints(edge,20,1)
#l_w = 20
def sample_random_Walks():
    #print("Running Random Walk, Length of edges, ", len(edges))
    random_walks = []
    for edge in edges:
        random_walks.append(run_random_walk_without_temporal_constraints(edge,l_w,1))
    #print("length of collected random walks,", len(random_walks))
    random_walks = [item for item in random_walks if item is not None]
    #print("length of collected random walks after removing None,", len(random_walks))
    random_walks = [clean_random_walk(item) for item in random_walks]
    random_walks = [item for item in random_walks if filter_rw(item,filter_walk)]
    #print("Length of random walks after removing short ranadom walks", len(random_walks))
    return random_walks
def get_sequences_from_random_walk(random_walks):
    sequences = [convert_walk_to_seq(item) for item in random_walks]
    sequences = [convert_seq_to_id(vocab, item) for item in sequences]
    sequences = [get_time_delta(item,0) for item in sequences]
    return sequences
random_walks = sample_random_Walks()
sequences = get_sequences_from_random_walk(random_walks)
#print("Average length of random walks")
lengths = []
for wk in random_walks:
    lengths.append(len(wk))
print("Mean length {} and Std deviation {}".format(str(np.mean(lengths)),str(np.std(lengths))))

inter_times = []
for seq in sequences:
    for item in seq:
        inter_times.append(np.log(item[2]))
mean_log_inter_time = np.mean(inter_times)
std_log_inter_time = np.std(inter_times)
print("mean log inter time and std log inter time ", mean_log_inter_time,std_log_inter_time)


import os
import json
import pickle
start_node_and_times = [(seq[0][0],seq[0][1],seq[0][2]) for seq in sequences ]
config_dir = config_path ### Change in random walks
isdir = os.path.isdir(config_dir) 
if not isdir:
    os.mkdir(config_dir)
isdir = os.path.isdir(config_dir+"/models") 
if not isdir:
    os.mkdir(config_dir+"/models")
pickle.dump({"mean_log_inter_time":mean_log_inter_time,"std_log_inter_time":std_log_inter_time},open(config_dir+"/time_stats.pkl","wb"))
pickle.dump(vocab,open(config_dir+"/vocab.pkl","wb"))

def evaluate_model(elstm):
    elstm.eval()
    running_loss= []
    running_event_loss = []
    running_time_loss= []
    event_prediction_rates = []
    topk_event_prediction_rates = []
    topk10_event_prediction_rates = []
    topk20_event_prediction_rates = []

    random_walks = sample_random_Walks()
    sequences = get_sequences_from_random_walk(random_walks)
    #start_node_and_times += [(seq[0][0],seq[0][1],seq[0][2]) for seq in sequences ]

    seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len= get_X_Y_T_from_sequences(sequences)
    seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths = data_shuffle(seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths)

    for start_index in range(0, len(seq_X),batch_size):
        if start_index+batch_size < len(seq_X):
            #print(start_index,batch_size,len(seq_X))
            try:
                pad_batch_X,pad_batch_Y,pad_batch_Xt,pad_batch_Yt,pad_batch_XDelta,pad_batch_YDelta,batch_X_len,batch_Y_len = get_batch(start_index,batch_size,seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths)
                mask_distribution = pad_batch_Y!=0
                num_events_time_loss = mask_distribution.sum().item()

                Y_hat,inter_time_log_loss = elstm(X=pad_batch_X,Y=pad_batch_Y,
                                Xt=pad_batch_Xt,Yt = pad_batch_Yt,
                                XDelta = pad_batch_XDelta,YDelta = pad_batch_YDelta,
                                X_lengths= batch_X_len,mask=mask_distribution)
                Y = pad_batch_Y
                Y = Y.view(-1)
                Y = Y-1  
                Y_hat = Y_hat.view(-1, Y_hat.shape[-1])      
                event_prediction_rates.append(get_event_prediction_rate(Y,Y_hat))
                topk_event_prediction_rates.append(get_topk_event_prediction_rate(Y,Y_hat,k=5))
                topk10_event_prediction_rates.append(get_topk_event_prediction_rate(Y,Y_hat,k=10))
                topk20_event_prediction_rates.append(get_topk_event_prediction_rate(Y,Y_hat,k=20))
            except:
                print("Error encountered")
    print("Event prediction rate:,", np.mean(event_prediction_rates))
    print("Event prediction rate@top5:,", np.mean(topk_event_prediction_rates))
    #print("Event prediction rate@top10:,", np.mean(topk10_event_prediction_rates))
    #print("Event prediction rate@top20:,", np.mean(topk20_event_prediction_rates))
def get_X_Y_T_from_sequences(sequences):
    seq_X = []
    seq_Y = []
    seq_Xt = []
    seq_Yt = []
    seq_XDelta = []
    seq_YDelta = []
    for seq in sequences:
        seq_X.append([item[0] for item in seq[:-1]])  ## O contain node id
        seq_Y.append([item[0] for item in seq[1:]])
        seq_Xt.append([item[1] for item in seq[:-1]])   ## 1 contain timestamp
        seq_Yt.append([item[1] for item in seq[1:]])
        seq_XDelta.append([item[2] for item in seq[:-1]])   ## 2 contain delta from previous event
        seq_YDelta.append([item[2] for item in seq[1:]])
    X_lengths = [len(sentence) for sentence in seq_X]
    Y_lengths = [len(sentence) for sentence in seq_Y]
    max_len = max(X_lengths)
    return seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len

def get_batch(start_index,batch_size,seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths):
    batch_X = seq_X[start_index:start_index+batch_size]
    batch_Y = seq_Y[start_index:start_index+batch_size]
    batch_Xt = seq_Xt[start_index:start_index+batch_size]
    batch_Yt = seq_Yt[start_index:start_index+batch_size] 
    batch_XDelta = seq_XDelta[start_index:start_index+batch_size]
    batch_YDelta = seq_YDelta[start_index:start_index+batch_size]   
    batch_X_len = X_lengths[start_index:start_index+batch_size]
    batch_Y_len = Y_lengths[start_index:start_index+batch_size]
    max_len = max(batch_X_len)

    pad_batch_X = np.ones((batch_size, max_len),dtype=np.int64)*pad_token
    pad_batch_Y = np.ones((batch_size, max_len),dtype=np.int64)*pad_token
    pad_batch_Xt = np.ones((batch_size, max_len),dtype=np.float32)*pad_token
    pad_batch_Yt = np.ones((batch_size, max_len),dtype=np.float32)*pad_token
    pad_batch_XDelta = np.ones((batch_size, max_len),dtype=np.float32)*pad_token
    pad_batch_YDelta = np.ones((batch_size, max_len),dtype=np.float32)*pad_token

    for i, x_len in enumerate(batch_X_len):
        pad_batch_X[i, 0:x_len] = batch_X[i][:x_len]
        pad_batch_Y[i, 0:x_len] = batch_Y[i][:x_len]
        pad_batch_Xt[i, 0:x_len] = batch_Xt[i][:x_len]
        pad_batch_Yt[i, 0:x_len] = batch_Yt[i][:x_len]
        pad_batch_XDelta[i, 0:x_len] = batch_XDelta[i][:x_len]
        pad_batch_YDelta[i, 0:x_len] = batch_YDelta[i][:x_len]

    pad_batch_X =  torch.LongTensor(pad_batch_X).to(device)
    pad_batch_Y =  torch.LongTensor(pad_batch_Y).to(device)
    pad_batch_Xt =  torch.Tensor(pad_batch_Xt).to(device)
    pad_batch_Yt =  torch.Tensor(pad_batch_Yt).to(device)
    pad_batch_XDelta =  torch.Tensor(pad_batch_XDelta).to(device)
    pad_batch_YDelta =  torch.Tensor(pad_batch_YDelta).to(device)
    batch_X_len = torch.LongTensor(batch_X_len).to(device)
    batch_Y_len = torch.LongTensor(batch_Y_len).to(device)
    return pad_batch_X,pad_batch_Y,pad_batch_Xt,pad_batch_Yt,pad_batch_XDelta,pad_batch_YDelta,batch_X_len,batch_Y_len

def data_shuffle(seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths):
    indices = list(range(0, len(seq_X)))
    random.shuffle(indices)
    #### Data Shuffling
    seq_X = [seq_X[i] for i in indices]   #### 
    seq_Y = [seq_Y[i] for i in indices]
    seq_Xt = [seq_Xt[i] for i in indices]
    seq_Yt = [seq_Yt[i] for i in indices]    
    seq_XDelta = [seq_XDelta[i] for i in indices]
    seq_YDelta = [seq_YDelta[i] for i in indices]
    X_lengths = [X_lengths[i] for i in indices]
    Y_lengths = [Y_lengths[i] for i in indices]
    return seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths


seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len= get_X_Y_T_from_sequences(sequences)
print("Max lengths of walks", max_len)
seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths = data_shuffle(seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths)

if gpu_num == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(str(gpu_num)) if torch.cuda.is_available() else "cpu")

print("Computation device, ", device)



batch_size = 128  ### Experiment wit 

elstm = EventLSTM(vocab=vocab,nb_layers=2, nb_lstm_units=200,time_emb_dim= 64,
    embedding_dim=100, batch_size= batch_size,device=device,
    mean_log_inter_time=mean_log_inter_time,
    std_log_inter_time=std_log_inter_time)
elstm = elstm.to(device)
celoss = nn.CrossEntropyLoss(ignore_index=-1) #### -1 is padded     
optimizer = optim.Adam(elstm.parameters(), lr=.001)
num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)
print(" ##### Number of parameters#### " ,num_params)

#elstm.train()
print_ct = 100000
wt_update_ct = 0
debug = False
for epoch in range(0,num_epochs+1):
    print("\r%d/%d"%(epoch,num_epochs),end="")
    elstm.train()
    #print("Epoch :",epoch)
    running_loss= []
    running_event_loss = []
    running_time_loss= []
    event_prediction_rates = []
    topk_event_prediction_rates = []
    topk10_event_prediction_rates = []
    random_walks = sample_random_Walks()
    sequences = get_sequences_from_random_walk(random_walks)
    start_node_and_times += [(seq[0][0],seq[0][1],seq[0][2]) for seq in sequences ]

    seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len= get_X_Y_T_from_sequences(sequences)
    seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths = data_shuffle(seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths)

    for start_index in range(0, len(seq_X),batch_size):
        if start_index+batch_size < len(seq_X):
            pad_batch_X,pad_batch_Y,pad_batch_Xt,pad_batch_Yt,pad_batch_XDelta,pad_batch_YDelta,batch_X_len,batch_Y_len = get_batch(start_index,batch_size,seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths)

            elstm.zero_grad()
            mask_distribution = pad_batch_Y!=0
            num_events_time_loss = mask_distribution.sum().item()

            Y_hat,inter_time_log_loss = elstm(X=pad_batch_X,Y=pad_batch_Y,
                            Xt=pad_batch_Xt,Yt = pad_batch_Yt,
                            XDelta = pad_batch_XDelta,YDelta = pad_batch_YDelta,
                            X_lengths= batch_X_len,mask=mask_distribution)
            
            inter_time_log_loss *= mask_distribution
            loss_time = (-1)*inter_time_log_loss.sum()*1.00/num_events_time_loss
            Y = pad_batch_Y
            Y = Y.view(-1)
            Y = Y-1  #### since index will start from 0 and also we have asked cross entropy loss to ignore -1 term
            # flatten all predictions
            Y_hat = Y_hat.view(-1, Y_hat.shape[-1])
            loss_event = celoss(Y_hat,Y)

            loss = loss_event+loss_time
            loss.backward()
            torch.nn.utils.clip_grad_norm_(elstm.parameters(), 0.5)
            optimizer.step()
            running_loss.append(loss.item())
            running_event_loss.append(loss_event.item())
            running_time_loss.append(loss_time.item())        

            wt_update_ct += 1

            if wt_update_ct%print_ct == 0 and debug:
                print("Running Loss :, ",np.mean(running_loss[-print_ct:]) )
                print("Running event loss: ", np.mean(running_event_loss[-print_ct:]),np.std(running_event_loss[-print_ct:]))
                print("Running time log loss: ", np.mean(running_time_loss[-print_ct:]),np.std(running_time_loss[-print_ct:]))


    #print("Epoch done")
    #print("Running Loss :, ",np.mean(running_loss) )
    #print("Running event loss: ", np.mean(running_event_loss),np.std(running_event_loss))
    #print("Running time log loss: ", np.mean(running_time_loss),np.std(running_time_loss))
    if epoch%20 == 0:
        print("Running Loss :, ",np.mean(running_loss) )
        print("Running event loss: ", np.mean(running_event_loss),np.std(running_event_loss))
        print("Running time log loss: ", np.mean(running_time_loss),np.std(running_time_loss))
        print("Running evaluation")
        evaluate_model(elstm)
        torch.save(elstm.state_dict(), config_dir+"/models/{}.pth".format(str(epoch)))
        import h5py
        hf = h5py.File(config_dir+'/start_node_and_times.h5', 'w')
        hf.create_dataset('1', data=start_node_and_times)
        hf.close()
    #print("")
