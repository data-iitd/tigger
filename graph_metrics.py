from metrics.metric_utils import get_numpy_matrix_from_adjacency
from metrics.metrics import compute_graph_statistics,create_timestamp_edges,calculate_temporal_katz_index,Edge
import sklearn
import argparse
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
def get_edges_from_adj_graph(graph):
    s = set()
    for start,adj_list in graph.items():
        for end,value in adj_list.items():
            if value> 0:
                start,end = min(start,end),max(start,end)
            s.add("_".join([str(start),str(end)]))
    return s

import math
parser = argparse.ArgumentParser()
parser.add_argument("--op",help="path of original graphs", type=str)
parser.add_argument("--sp",help="path of sampled graphs", type=str)
parser.add_argument('--time_window',default=1, help="Size of each time window where data needs to be generated",type=int)
parser.add_argument('--debug',default=0, help="debugger",type=int)
parser.add_argument('--config_name',default='', help="name of the config",type=str)

args = parser.parse_args()
f = open("results.txt",'a')
print(args)
original_graphs_path = args.op
sampled_graphs_path = args.sp
time_window = args.time_window
debug = args.debug
config_name = args.config_name
#outputfile = args.outputfile
original_graphs = pickle.load(open(original_graphs_path,"rb"))
sampled_graphs = pickle.load(open(sampled_graphs_path,"rb"))
print("Length of original and sampled,", len(original_graphs),len(sampled_graphs))
commons = []
for i in range(0, len(sampled_graphs)):
    sgraph = sampled_graphs[i]
    ograph = original_graphs[i]
    sgraphedges = get_edges_from_adj_graph(sgraph)
    ographedges = get_edges_from_adj_graph(ograph)
    len_o = len(ographedges)
    len_common = len(ographedges.intersection(sgraphedges))
    if len_o != 0:
        commons.append(len_common*100.0/len_o)
#     if len_o != 0:
#         print(i, )
mean_edge_intersection = np.mean(commons)
median_edge_intersection = np.median(commons)

result = {}
result['edge_diversity'] = np.median(commons)


df_metric = []
df_val = []


df_metric.append("mean_edge_overlap")
df_val.append(mean_edge_intersection)
df_metric.append("median_edge_overlap")
df_val.append(median_edge_intersection)

labels = []
old_stats = []
new_stats = []
ct = 0
labels = []
mmd_stats = []

import warnings;
warnings.filterwarnings('ignore');
for ct in range(len(sampled_graphs)):
    print("\r%d"%ct,end="")
    original_matrix,_,_ = get_numpy_matrix_from_adjacency(original_graphs[ct])
    sampled_matrix,_,_= get_numpy_matrix_from_adjacency(sampled_graphs[ct])

    assert ((original_matrix == original_matrix.T).all())
    assert ((sampled_matrix == sampled_matrix.T).all())
    if original_matrix.shape[0] > 10:
        labels.append(ct)
        old_graph_stats = compute_graph_statistics(original_matrix)
        new_graph_stats = compute_graph_statistics(sampled_matrix)
        old_stats.append(old_graph_stats)
        new_stats.append(new_graph_stats)
        if debug:
            print("original, sampled,", np.sum(original_matrix>0)*0.5,original_matrix.shape[0],np.sum(sampled_matrix>0)*0.5,sampled_matrix.shape[0])
        
        
        
import math
actual_graph_result = {}
for metric in old_stats[0].keys():
    actual_graph_metrics = [item[metric] for item in old_stats]
    sampled_graph_metrics = [item[metric] for item in new_stats]
    abs_error = [abs(a-b)*1.00 for a,b in zip(actual_graph_metrics,sampled_graph_metrics)]
    infs = [item for item in abs_error if ( pd.isnull(item) or  math.isinf(item)) ]
    if len(infs) >0:
        print("infs found, ", len(infs), metric)
    abs_error = [item for item in abs_error if ( not pd.isnull(item) and not math.isinf(item)) ]
    actual_graph_metrics = [item for item in actual_graph_metrics if ( not pd.isnull(item) and not math.isinf(item)) ]
    print("Actual graph metrices",len(actual_graph_metrics))
    result["{}".format(metric)] = np.median(abs_error)
    actual_graph_result["{}".format(metric)] = np.median(actual_graph_metrics)
    

print(result)
nums = []
for metric in ['edge_diversity',
               'd_mean','wedge_count','triangle_count','power_law_exp',
                'rel_edge_distr_entropy','LCC','n_components','clustering_coefficient',
                'betweenness_centrality_mean',
                'closeness_centrality_mean']:
    nums.append(result[metric])

results = [np.round(item,4) for item in nums]
print("median")
nums = "& ".join(["$"+str(item)+"$" for item in results])
print(nums)
# print("dumping result")








