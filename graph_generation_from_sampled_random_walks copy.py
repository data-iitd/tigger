#%%
import argparse
import os
from metrics.metric_utils import get_numpy_matrix_from_adjacency,get_adj_graph_from_random_walks,get_total_nodes_and_edges_from_temporal_adj_list_in_time_range,get_adj_origina_graph_from_original_temporal_graph
from metrics.metric_utils import sample_adj_graph_multinomial_k_inductive,sample_adj_graph_topk
from tgg_utils import *
import pandas as pd
import pickle
from collections import defaultdict
import sys
#%%
### configurations 
data_path = "data/bitcoin/data.csv"
config_dir = "temp/bitcoin_org"
num_of_sampled_graphs = 1
time_window = 1
topk_edge_sampling= None
l_w = 20

#%%

strictly_increasing_walks = True
num_next_edges_to_be_stored = 100
undirected = True

data= pd.read_csv(data_path)

data = data[['start','end','days']]
node_set = set(data['start']).union(set(data['end']))
#print("number of nodes,",len(node_set))
node_set.update('end_node')
max_days = max(data['days'])
#print("Minimum, maximum timestamps",min(data['days']),max_days)
data = data.sort_values(by='days',inplace=False)
# print("number of interactions," ,data.shape[0])
# print(data.head())


vocab = pickle.load(open(config_dir+"/vocab.pkl","rb"))
reverse_vocab = {value:key for key, value in vocab.items()}
end_node_id = vocab['end_node']

#%%

temporal_graph_original = defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: 0)))
for start,end,day in data[['start','end','days']].values:
    temporal_graph_original[day][start][end] += 1
    if undirected:
        temporal_graph_original[day][end][start] += 1

target_node_counts = []
target_edge_counts = []
time_labels = []
for start_time in range(1,max_days,time_window):
    # determine node and edge count per time frame
    tp,node_count = get_total_nodes_and_edges_from_temporal_adj_list_in_time_range(temporal_graph_original,start_time,start_time+time_window-1)
    if undirected:
        tp = int(tp/2)
    
    target_edge_counts.append(tp)
    target_node_counts.append(node_count)
#print(target_edge_counts,target_node_counts)
#%%
original_graphs = []
# get subgraph for time label
for start_time in range(1,max_days,time_window):
    time_labels.append(start_time)
    original_graphs.append(get_adj_origina_graph_from_original_temporal_graph(temporal_graph_original,start_time,start_time+time_window-1))
degree_distributions = []
# determine edge degree per node per time frame
for i,graph in enumerate(original_graphs):
    temp,_,_ = get_numpy_matrix_from_adjacency(graph)  # adj matrix
    degree_distributions.append(list(temp.sum(axis=0)))  # sum adj matrix = degree
   
#%% 
import h5py
pickle.dump(original_graphs,open(config_dir+"/results/original_graphs.pkl","wb"))
pickle.dump(time_labels,open(config_dir+"/results/time_labels.pkl","wb"))
pickle.dump(max_days,open(config_dir+"/results/max_days.pkl","wb"))
def sequences_from_temporal_walks(generated_events,generated_times):
    sampled_walks = []
    lengths =[]
    for i in range(generated_times.shape[0]):
        sample_walk_event = []
        sample_walk_time = []
        done = False
        j = 0
        while not done and j <= l_w:  # loop throught random walk
            event = generated_events[i][j]
            time = generated_times[i][j]
            j += 1
            if event == 1 or time > max_days:
                done = True
            else:
                sample_walk_event.append(reverse_vocab[event])
                sample_walk_time.append(time)
        lengths.append(len(sample_walk_event))
        sampled_walks.append((sample_walk_event,sample_walk_time))
    print("Mean length {} and Std deviation {}".format(str(np.mean(lengths)),str(np.std(lengths))))
    sampled_walks = [item for item in sampled_walks if len(item[0]) >= 3]
    print(len(sampled_walks))
    return sampled_walks

#%%

list_of_sampled_walks = []
for i in range(0,num_of_sampled_graphs):
    print(i)
    generated_events = np.load(open(config_dir+"/results/generated_events_{}.npy".format(str(i)),"rb"))
    generated_times = np.load(open(config_dir+"/results/generated_times_{}.npy".format(str(i)),"rb"))
    print(i,generated_events.shape[0],generated_times.shape[0])
    sampled_walks = sequences_from_temporal_walks(generated_events,generated_times)
    # convert walk into a adjacencly list
    adj_matrix_temporal_sampled = get_adj_graph_from_random_walks(sampled_walks,1,max_days,True)
    sampled_graphs = []
    ct = 0
    for start_time in range(3,max_days,time_window):
        print("\r%d/%d"% (ct,start_time),end="")
        if topk_edge_sampling:
            sampled_graphs.append(sample_adj_graph_topk(
                adj_matrix_temporal_sampled,
                start_time,
                start_time+time_window-1,
                target_edge_counts[ct],
                target_node_counts[ct],
                degree_distributions[ct],
                True)
                )
        else:
            sampled_graphs.append(sample_adj_graph_multinomial_k_inductive(
                adj_matrix_temporal_sampled,
                start_time,
                start_time+time_window-1,
                target_edge_counts[ct],
                target_node_counts[ct],
                degree_distributions[ct],
                True))
        #print("original, sampled,", np.sum(original_matrix>0)*0.5,original_matrix.shape[0],np.sum(sampled_matrix>0)*0.5,sampled_matrix.shape[0])

        ct += 1 
    fp = open(config_dir+"/results/sampled_graph_{}.pkl".format(str(i)),"wb")
    pickle.dump(sampled_graphs,fp)    
    fp.close()
    print("Dumped the generated graph\n")
os._exit(os.EX_OK)

#sys.exit(0)
    


    

# %%
