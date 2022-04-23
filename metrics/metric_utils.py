from collections import Counter,defaultdict
from tqdm import tqdm
import numpy as np
def get_string_from_edge_tuple(start_node,end_node,time,connector="#"):
    return connector.join([str(start_node),str(end_node),str(time)])

def get_total_edges(adj_matrix):
    ct = 0
    for key, value in adj_matrix.items():
        for end, edge in value.items():
            if edge>= 1:
                ct += 1
    return ct

def convert_graph_from_defauldict_to_dict(graph):   #### Assuming each key contains a defaultdict graph too
    for key,_ in graph.items():
        graph[key] = dict(graph[key])
    return dict(graph)
def get_adj_graph_from_random_walks(random_walks,start_time, end_time,undirected=True):
    graph = defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: 0)))
    for wk in tqdm(random_walks):
        events = wk[0]
        times = wk[1]
        edges = []
        done = False
        for index,time in enumerate(times):
            if index >= 1 and not done:
                if time >= start_time and time <= end_time:
                    start_node = events[index-1]
                    end_node = events[index]
                    graph[time][start_node][end_node] += 1
                    if undirected:
                        graph[time][end_node][start_node] += 1
    
    return convert_graph_from_defauldict_to_dict(graph)

def aggregate_adj_matrices(graphs):
    graph = graphs[0].copy()
    for ng in graphs[1:]:
        for start_node, adj_list in ng.items():
            if start_node not in graph:
                graph[start_node] = defaultdict(lambda :0)
            for end_node,count in adj_list.items():
                graph[start_node][end_node] += count
    return graph

def get_numpy_matrix_from_adjacency(adj,keep_weights= False):
    num_nodes = len(adj)
    node_to_id = {}
    for nd in adj.keys():
        node_to_id[nd] = len(node_to_id)
    matrix = np.zeros((num_nodes,num_nodes))
    for key , value in adj.items():
        for key2, value2 in value.items():
            if not keep_weights and value2 > 0:
                value2 = 1
            matrix[node_to_id[key]][node_to_id[key2]] = value2
    #num_edges_actual = int(np.sum(matrix> 0)/2)
    matrix = matrix.astype(int)
    return matrix,node_to_id,{value:key for key,value in node_to_id.items()}


def get_unique_string_from_edge_tuple(start_node,end_node,time):
    if start_node > end_node:
        start_node,end_node = end_node,start_node
    return get_string_from_edge_tuple(start_node,end_node,time)


def get_binary_adj_graph_from_sampled_temporal_adj_matrix(graph,start,end,edge_count_needed,undirected=True):
    required_counts = edge_count_needed
    print(required_counts)
    tedges = {}
    for time in range(start,end+1):
        for start_node, adj_list in graph[time].items():
            for end_node, count in adj_list.items():
                if start_node != end_node:
                    if undirected:
                        key = get_unique_string_from_edge_tuple(start_node,end_node,time)
                    else:
                        key = get_string_from_edge_tuple(start_node,end_node,time)
                    if key in tedges:
                        tedges[key] += count
                    else:
                        tedges[key] = count
                        

    edges = {}
    for edge,ct in tedges.items():
        edge = edge.split("#")
        start_node,end_node = edge[0],edge[1]
        key = "#".join([start_node,end_node])
        if key not in edges:
            edges[key] = ct
        else:
            edges[key] += ct
    counts = list(edges.values())
    counts.sort(reverse=True)
    threshold = counts[required_counts] + 1
    
    #print(sum([1 for item in counts if item >= threshold]))
    #print(threshold)
    #graph = defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: 0)))
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    for edge,ct in edges.items():
        if ct >= threshold:
            start_node,end_node = edge.split("#")
            graph[start_node][end_node] = 1
            if undirected:
                graph[end_node][start_node] = 1
    print(get_total_edges(graph)*.5)
    return graph
    
    
def get_adj_origina_graph_from_original_temporal_graph(temporal_original_graph,start_time,end_time):
    graph = defaultdict(lambda : defaultdict(lambda: 0))
    for time in range(start_time,end_time+1):
        #print(time)
        for start_node,adj_list in temporal_original_graph[time].items():
            for end_node, value in adj_list.items():
                if value > 0:
                    graph[start_node][end_node] = 1
    for key,value in graph.items():
        graph[key] = dict(graph[key]) ### converting from defaultdict to dict
    return dict(graph)
def get_total_edges_from_temporal_adj_list_in_time_range(temporal_adj_matrix,start_time,end_time):
    ct = 0
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    for time in range(start_time,end_time+1,1):
        for start_node, adj_list in temporal_adj_matrix[time].items():
            for end_node, value in adj_list.items():
                if value>0:
                    graph[start_node][end_node] = 1
    return get_total_edges(graph)    

def get_total_nodes_and_edges_from_temporal_adj_list_in_time_range(temporal_adj_matrix,start_time,end_time):
    ct = 0
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    unique_nodes = set()
    for time in range(start_time,end_time+1,1):
        for start_node, adj_list in temporal_adj_matrix[time].items():
            unique_nodes.add(start_node)
            for end_node, value in adj_list.items():
                if value>0:
                    graph[start_node][end_node] = 1
                    unique_nodes.add(end_node)
    return get_total_edges(graph),len(unique_nodes) 
def get_graph_degree_sequence(graph):
    current_deg_seq_dict = defaultdict(lambda :0)
    for start,adj_list in graph.items():
        for end,ct in adj_list.items():
            current_deg_seq_dict[start] += 1
    current_deg_seq = [(key,value) for key,value in current_deg_seq_dict.items()]
    current_deg_seq.sort(reverse=True,key=lambda val:val[1])
    return current_deg_seq_dict,current_deg_seq
def get_edges_nodes_graph(graph):
    node_ct = 0
    edge_ct = 0
    for start,adj_list in graph.items():
        if len(adj_list) >0:
            node_ct += 1
        for end,ct in adj_list.items():
            if ct > 0:
                edge_ct += 1
    return node_ct,edge_ct/2


def get_binary_adj_graph_from_sampled_temporal_adj_matrix_using_node_wise_counts(graph,start,end,edge_count_needed,node_count_needed,undirected=True):
    required_edge_counts = edge_count_needed
    print("Required edges count and node count , " ,required_edge_counts,node_count_needed)
    
#     ### first figure out top nodes which have lots of edges.
#     node_ct_dict = defaultdict(lambda :0)
#     for time in range(start,end+1):
#         for start_node, adj_list in graph[time].items():
#             for end_node, count in adj_list.items():
#                 if start_node != end_node:
#                     node_ct_dict[start_node] += 1
#                     node_ct_dict[end_node] += 1
                    
#     counts = list(node_ct_dict.values())
#     counts.sort(reverse=True)
#     if node_count_needed < len(counts):
#         threshold = counts[node_count_needed] + 1
#     else:
#         threshold = counts[-1]
#     required_nodes = set()
#     for key, ct in node_ct_dict.items():
#         if ct >= threshold:
#             required_nodes.add(key)

    #print("Length of edge needed and nodes needed ,", required_edge_counts,len(required_nodes) , " ")

    tedges = {}
    for time in range(start,end+1):
        for start_node, adj_list in graph[time].items():
            for end_node, count in adj_list.items():
                if start_node != end_node: #and start_node in required_nodes and end_node in required_nodes :
                    if undirected:
                        key = get_unique_string_from_edge_tuple(start_node,end_node,time)
                    else:
                        key = get_string_from_edge_tuple(start_node,end_node,time)
                    if key in tedges:
                        tedges[key] += count
                    else:
                        tedges[key] = count
                        

    edges = {}
    for edge,ct in tedges.items():
        edge = edge.split("#")
        start_node,end_node = edge[0],edge[1]
        key = "#".join([start_node,end_node])
        if key not in edges:
            edges[key] = ct
        else:
            edges[key] += ct
    counts = list(edges.values())
    counts.sort(reverse=True)
    print("Length of extracted edges", len(counts))
    if required_edge_counts < len(counts):
        threshold = counts[required_edge_counts]
    else:
        threshold = counts[-1]-1
    edges = [(key,ct) for key,ct in edges.items()]
    edges.sort(key=lambda x:x[1],reverse=True)

    graph = defaultdict(lambda:defaultdict(lambda: 0))
    final_nodes = set()
    final_edge_ct = 0
    index = 0
    done = False
    for edge,ct in edges:
        if ct >= threshold and not done:
            index += 1
            start_node,end_node = edge.split("#")
            graph[start_node][end_node] = 1
            if undirected:
                graph[end_node][start_node] = 1
            final_nodes.add(start_node)
            final_nodes.add(end_node)
            final_edge_ct += 1
            if final_edge_ct >= required_edge_counts:
                done = True
    print("Acquired edges and nodes till first stage", get_total_edges(graph)*.5, len(final_nodes))
#     #print(index,len(final_nodes))
#     newindex = index
#     for edge,ct in edges[index:]:
#         newindex += 1
#         start_node,end_node = edge.split("#")
#         if final_edge_ct*100/required_counts <=100 :#or len(final_nodes) < node_count_needed:
#             #if coin_toss(.5):
#             if start_node in final_nodes and end_node in final_nodes:
#                 graph[start_node][end_node] = 1
#                 if undirected:
#                     graph[end_node][start_node] = 1
#                 final_nodes.add(start_node)
#                 final_nodes.add(end_node)
#                 final_edge_ct += 1  
# #             else:
# #                 graph[start_node][end_node] = 1
# #                 if undirected:
# #                     graph[end_node][start_node] = 1
# #                 final_nodes.add(start_node)
# #                 final_nodes.add(end_node)
# #                 final_edge_ct += 1                     
            
#     print("Acquired edges and nodes till this stage", get_total_edges(graph)*.5, len(final_nodes))
#     print("Index reached ", newindex)
    return graph
    
    
    
    
    
def sample_adj_graph_havel_hakimmi(graph,start,end,edge_count_needed,node_count_needed,req_deg_seq,undirected=True):
    req_edge_ct = edge_count_needed
    req_node_ct = node_count_needed
    print("Required edges count and node count , " ,req_edge_ct,req_node_ct)
    req_deg_seq.sort(reverse=True)

    tedges = {}
    for time in range(start,end+1):
        for start_node, adj_list in graph[time].items():
            for end_node, count in adj_list.items():
                if start_node != end_node: 
                    if undirected:
                        key = get_unique_string_from_edge_tuple(start_node,end_node,time)
                    else:
                        key = get_string_from_edge_tuple(start_node,end_node,time)
                    if key in tedges:
                        tedges[key] += count
                    else:
                        tedges[key] = count
                        
    #print(tedges)
    edges = {}
    node_interaction_ct = defaultdict(lambda:defaultdict(lambda: 0))
    for edge,ct in tedges.items():
        edge = edge.split("#")
        start_node,end_node = edge[0],edge[1]
        key = "#".join([start_node,end_node])
        node_interaction_ct[start_node][end_node] += ct
        node_interaction_ct[end_node][start_node] += ct
        if key not in edges:
            edges[key] = ct
        else:
            edges[key] += ct


    counts = list(edges.values())
    counts.sort(reverse=True)
    print("Length of extracted edges", len(counts))
    if req_edge_ct < len(counts):
        threshold = counts[req_edge_ct]
    else:
        threshold = counts[-1]-1
    edges = [(key,ct) for key,ct in edges.items()]
    edges.sort(key=lambda x:x[1],reverse=True)

    for node,adj_list in node_interaction_ct.items():
        tp= [(key,value) for key,value in adj_list.items()]
        tp.sort(reverse=True,key=lambda val: val[1])
        node_interaction_ct[node] = tp
        
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    final_nodes = set()
    final_edge_ct = 0
    index = 0
    done = False
    for edge,ct in edges:
        if ct >= threshold and not done:
            index += 1
            start_node,end_node = edge.split("#")
            graph[start_node][end_node] = ct
            if undirected:
                graph[end_node][start_node] = ct
            final_nodes.add(start_node)
            final_nodes.add(end_node)
            final_edge_ct += 1
            if final_edge_ct >= req_edge_ct:
                done = True
                
    ### This is the tentative graph but lets clean it further  ###
        
    ### Now clean this graph using Havel Hakimmi algorithm
    ### first figure out the current degree sequence and sort it
    ### Then add/delete nbrs based on the degree sequence requirement
    ### finally get the new degree sequence requirement
    ### Now complete this requirement ###
    print("Required degree sequence")
    print(req_deg_seq,np.sum(req_deg_seq)/2)
    current_deg_seq_dict,current_deg_seq = get_graph_degree_sequence(graph)
    print("After first step")
    print([item[1] for item in current_deg_seq],np.sum([item[1] for item in current_deg_seq])/2)
    #print(len(current_deg_seq),np.sum([item[1] for item in current_deg_seq])/2)
    print("nodes and edges", get_edges_nodes_graph(graph))
    #print(current_deg_seq)
    index = 0
    for node,ct in current_deg_seq:
        #print(index)
        if index < len(req_deg_seq):
            req_deg = req_deg_seq[index]
            index += 1
            if ct > req_deg:
                edges_to_remove = []
                excess_nodes = ct - req_deg
                adj_list = graph[node]
                adj_nbrs = [(key,value) for key,value in adj_list.items()]
                adj_nbrs.sort(key=lambda val:val[1])
                for key,_ in adj_nbrs[:excess_nodes]:
                    del graph[node][key]
                    del graph[key][node]
        ## remove lowest possible excess_nodes
    updated_deg_seq_dict,updated_deg_seq = get_graph_degree_sequence(graph)
    #print("Required",req_deg_seq)
    #print(updated_deg_seq)
    print("After removing excessive nodes")
    print([item[1] for item in updated_deg_seq],np.sum([item[1] for item in updated_deg_seq])/2)
    #print(len(updated_deg_seq),np.sum([item[1] for item in updated_deg_seq])/2)
    print("nodes and edges", get_edges_nodes_graph(graph))
    cur_nodes_in_graph = set()
    for start,adj_list in graph.items():
        if len(adj_list) >0:
            cur_nodes_in_graph.add(start)
#     for node,_ in node_interaction_ct.items():
#         print(node_interaction_ct[node])
#     done_nodes = set()
#     for i,req_deg in enumerate(req_deg_seq):
#         if i < len(updated_deg_seq):
#             node = updated_deg_seq[i][0]
#             cur_degree = updated_deg_seq_dict[node] ## this will keep updating
#             if req_deg > cur_degree:

#                 req_deg = req_deg-cur_degree
#                 cur_nbrs = set([key for key,_ in graph[i].items()])
#                 interaction_nbrs = node_interaction_ct[node]
#                 #print(interaction_nbrs)
#                 add_ = 0
#                 for nbr,_ in interaction_nbrs:
                    
#                     if add_ < req_deg and nbr not in cur_nodes_in_graph and nbr not in done_nodes:
#                         add_ += 1
#                         graph[node][nbr] = 1
#                         graph[nbr][node] =1
#                         updated_deg_seq_dict[node] += 1
#                         updated_deg_seq_dict[nbr] += 1
#                 #print("added, ", add_)
#             done_nodes.add(node)
    print("After adding required edges")
    _,updated_deg_seq = get_graph_degree_sequence(graph)
    #print([item[1] for item in updated_deg_seq],np.sum([item[1] for item in updated_deg_seq])/2)
    print(len(updated_deg_seq),np.sum([item[1] for item in updated_deg_seq])/2)
    print("nodes and edges", get_edges_nodes_graph(graph))
    from copy import deepcopy
    old_graph = deepcopy(graph)
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    for start,adj_list in old_graph.items():
        for end, ct in adj_list.items():
            if ct >0:
                graph[start][end] = 1
    return graph





def sample_adj_graph_topk(graph,start,end,edge_count_needed,node_count_needed,req_deg_seq,undirected=True):
    req_edge_ct = edge_count_needed
    req_node_ct = node_count_needed
    #print("Required edges count and node count , " ,req_edge_ct,req_node_ct)
    req_deg_seq.sort(reverse=True)
    #print(req_deg_seq)
    tedges = {}
    for time in range(start,end+1):
        if time in graph:
            for start_node, adj_list in graph[time].items():
                for end_node, count in adj_list.items():
                    if start_node != end_node: 
                        if undirected:
                            key = get_unique_string_from_edge_tuple(start_node,end_node,time)
                        else:
                            key = get_string_from_edge_tuple(start_node,end_node,time)
                        if key in tedges:
                            tedges[key] += count
                        else:
                            tedges[key] = count
                        
    #print(tedges)
    edges = {}
    node_interaction_ct = defaultdict(lambda:defaultdict(lambda: 0))
    for edge,ct in tedges.items():
        edge = edge.split("#")
        start_node,end_node = edge[0],edge[1]
        key = "#".join([start_node,end_node])
        node_interaction_ct[start_node][end_node] += ct
        node_interaction_ct[end_node][start_node] += ct
        if key not in edges:
            edges[key] = ct
        else:
            edges[key] += ct


    counts = list(edges.values())
    counts.sort(reverse=True)
    #print("Length of extracted edges", len(counts))
    if req_edge_ct < len(counts):
        threshold = counts[req_edge_ct]
    else:
        threshold = counts[-1]-1
    edges = [(key,ct) for key,ct in edges.items()]
    edges.sort(key=lambda x:x[1],reverse=True)

    for node,adj_list in node_interaction_ct.items():
        tp= [(key,value) for key,value in adj_list.items()]
        tp.sort(reverse=True,key=lambda val: val[1])
        node_interaction_ct[node] = tp
        
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    final_nodes = set()
    final_edge_ct = 0
    index = 0
    done = False
    
    for edge,ct in edges:
        if ct >= threshold and not done:
            index += 1
            start_node,end_node = edge.split("#")
            graph[start_node][end_node] = ct
            if undirected:
                graph[end_node][start_node] = ct
            final_nodes.add(start_node)
            final_nodes.add(end_node)
            final_edge_ct += 1
            if final_edge_ct >= req_edge_ct:
                done = True
                
    ### This is the tentative graph but lets clean it further  ###
        
    ### Now clean this graph using Havel Hakimmi algorithm
    ### first figure out the current degree sequence and sort it
    ### Then add/delete nbrs based on the degree sequence requirement
    ### finally get the new degree sequence requirement
    ### Now complete this requirement ###
    #print("Required degree sequence")
    #print(req_deg_seq)
    current_deg_seq_dict,current_deg_seq = get_graph_degree_sequence(graph)

    from copy import deepcopy
    old_graph = deepcopy(graph)
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    for start,adj_list in old_graph.items():
        for end, ct in adj_list.items():
            if ct >0:
                graph[start][end] = 1
            
    return convert_graph_from_defauldict_to_dict(graph)
# start_time = 1
# ct = 0
# ggg = sample_adj_graph_havel_hakimmi(adj_matrix_temporal_sampled,start_time,start_time+window-1,target_edge_counts[ct],target_node_counts[ct],degree_distributions[ct],True)

    
# start_time = 1
# ct = 0
# ggg = sample_adj_graph_havel_hakimmi(adj_matrix_temporal_sampled,start_time,start_time+window-1,target_edge_counts[ct],target_node_counts[ct],degree_distributions[ct],True)

def sample_adj_graph_multinomial_k_edges(graph,start,end,edge_count_needed,node_count_needed,req_deg_seq,undirected=True,debug=False):
    req_edge_ct = edge_count_needed
    req_node_ct = node_count_needed
    if debug:
        print("Required edges count and node count , " ,req_edge_ct,req_node_ct)
    req_deg_seq.sort(reverse=True)
    #print(req_deg_seq)
    tedges = {}
    for time in range(start,end+1):
        if time in graph:
            for start_node, adj_list in graph[time].items():
                for end_node, count in adj_list.items():
                    if start_node != end_node: 
                        if undirected:
                            key = get_unique_string_from_edge_tuple(start_node,end_node,time)
                        else:
                            key = get_string_from_edge_tuple(start_node,end_node,time)
                        if key in tedges:
                            tedges[key] += count
                        else:
                            tedges[key] = count
                        
    #print(tedges)
    edges = {}
    node_interaction_ct = defaultdict(lambda:defaultdict(lambda: 0))
    for edge,ct in tedges.items():
        edge = edge.split("#")
        start_node,end_node = edge[0],edge[1]
        key = "#".join([start_node,end_node])
        node_interaction_ct[start_node][end_node] += ct
        node_interaction_ct[end_node][start_node] += ct
        if key not in edges:
            edges[key] = ct
        else:
            edges[key] += ct


    counts = list(edges.values())
    counts.sort(reverse=True)
    if debug:
        print("Length of extracted edges", len(counts))
    if req_edge_ct < len(counts):
        threshold = counts[req_edge_ct]
    else:
        threshold = counts[-1]-1
        
        
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    edges = [(key,ct) for key,ct in edges.items()]
    pairs = [a[0] for a in edges]
    wgts = [a[1] for a in edges]
    sum_ = np.sum(wgts)
    wgts = [a*1.00/sum_ for a in wgts]
    try:
        selected_edges = np.random.choice(pairs, req_edge_ct, p=wgts,replace=False)
    except:
        print("Error, sample size is greater than population")
        print("Selecting all the pairs")
        selected_edges = pairs
    #print(selected_edges[:100])
    for edge in selected_edges:
        start_node,end_node = edge.split("#")
        graph[start_node][end_node] = 1
        if undirected:
            graph[end_node][start_node] = 1
    

    if debug:
        print(req_deg_seq)
    current_deg_seq_dict,current_deg_seq = get_graph_degree_sequence(graph)
    if debug:
        print("After first step")
        print([item[1] for item in current_deg_seq])
        print("final nodes and edges", get_edges_nodes_graph(graph))

    if debug:
        print("final nodes and edges", get_edges_nodes_graph(graph))
    from copy import deepcopy
    old_graph = deepcopy(graph)
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    for start,adj_list in old_graph.items():
        for end, ct in adj_list.items():
            if ct >0:
                graph[start][end] = 1
    return convert_graph_from_defauldict_to_dict(graph)



def sample_adj_graph_multinomial_k_inductive(graph,start,end,edge_count_needed,node_count_needed,req_deg_seq,undirected=True,debug=False):
    req_edge_ct = edge_count_needed
    req_node_ct = node_count_needed
    if debug:
        print("Required edges count and node count , " ,req_edge_ct,req_node_ct)
    req_deg_seq.sort(reverse=True)
    #print(req_deg_seq)
    tedges = {}
    for time in range(start,end+1):
        if time in graph:
            for start_node, adj_list in graph[time].items():
                for end_node, count in adj_list.items():
                    if start_node != end_node: 
                        if undirected:
                            key = get_unique_string_from_edge_tuple(start_node,end_node,time)
                        else:
                            key = get_string_from_edge_tuple(start_node,end_node,time)
                        if key in tedges:
                            tedges[key] += count
                        else:
                            tedges[key] = count
                        
    #print(tedges)
    edges = {}
    node_interaction_ct = defaultdict(lambda:defaultdict(lambda: 0))
    for edge,ct in tedges.items():
        edge = edge.split("#")
        start_node,end_node = edge[0],edge[1]
        key = "#".join([start_node,end_node])
        node_interaction_ct[start_node][end_node] += ct
        node_interaction_ct[end_node][start_node] += ct
        if key not in edges:
            edges[key] = ct
        else:
            edges[key] += ct


    counts = list(edges.values())
    counts.sort(reverse=True)
    if debug:
        print("Length of extracted edges", len(counts))
    if req_edge_ct < len(counts):
        threshold = counts[req_edge_ct]
    else:
        threshold = counts[-1]-1
        
        
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    edges = [(key,ct) for key,ct in edges.items()]
    pairs = [a[0] for a in edges]
    wgts = [a[1] for a in edges]
    sum_ = np.sum(wgts)
    wgts = [a*1.00/sum_ for a in wgts]
    try:
        selected_edges = np.random.choice(pairs, req_edge_ct, p=wgts,replace=False)
    except:
        print("Sample size larger than population size")
        selected_edges = pairs
    #print(selected_edges[:100])
    for edge in selected_edges:
        start_node,end_node = edge.split("#")
        graph[start_node][end_node] = 1
        if undirected:
            graph[end_node][start_node] = 1
    

    if debug:
        print(req_deg_seq)
    current_deg_seq_dict,current_deg_seq = get_graph_degree_sequence(graph)
    if debug:
        print("After first step")
        print([item[1] for item in current_deg_seq])
        print("final nodes and edges", get_edges_nodes_graph(graph))

    if debug:
        print("final nodes and edges", get_edges_nodes_graph(graph))
    from copy import deepcopy
    old_graph = deepcopy(graph)
    graph = defaultdict(lambda:defaultdict(lambda: 0))
    for start,adj_list in old_graph.items():
        for end, ct in adj_list.items():
            if ct >0:
                graph[start][end] = 1
    return convert_graph_from_defauldict_to_dict(graph)



# def short_random_walk(wk,time_stamps_window):
#     newwk = []
#     start_time = wk[1][0]
#     i = 0
#     nodes = []
#     times = []
#     while i < len(wk[0]):
#         node = wk[0][i]
#         time = wk[1][i]
#         if time - start_time <= time_stamps_window:
#             nodes.append(node)
#             times.append(time)
#         i+= 1
#     return (nodes,times)