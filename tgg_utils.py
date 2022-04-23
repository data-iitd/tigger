import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
    
    
def sort_dict(diction):
    diction = [(key,value) for key, value in diction.items()]
    diction.sort(key=lambda val: val[1],reverse=True)
    return diction
#sort_dict(prods_new)
def print_incoming_outcoming_edges_of_edge(edge):
    
    print("Edge, ", edge)
    print("Incoming")
    for item in edge.incoming_edges:
        print(item)
        
    print("Outgoing")
    for item in edge.outgoing_edges:
        print(item)
    return

def sort_edges_timewise(edges,reverse):
    edges.sort(key= lambda val:val.time,reverse=reverse)
    return edges
def print_list_of_edges(edges,cut_off= 100):
    print("###")
    for index,edge in enumerate(edges):
        if index < cut_off:
            print(index,edge)
    print("###")
    
def prepare_alias_table(edge,incoming = False,window_interactions = 10):
    if not incoming:
        time_diffs = [item.time -edge.time for item in edge.outgoing_edges]
    else:
        time_diffs = [item.time -edge.time for item in edge.incoming_edges]
    mn = np.mean(time_diffs)
    std = np.std(time_diffs)
    if len(time_diffs) == 1 or std == 0:
        std = 1
        #print(time_diffs,[-(item - mn)/std for item in time_diffs] )
    time_diffs = [-(item - mn)/std for item in time_diffs]   ### less time diff edge should be more prioritized
    time_diffs = np.exp(time_diffs)
    norm_const = sum(time_diffs)
    nbr_sample_probs = [float(prob)/norm_const for prob in time_diffs]
    J, q  = alias_setup(nbr_sample_probs)
    return nbr_sample_probs,J,q

def print_incoming_outcoming_edges_of_edge(edge):
    
    print("Edge, ", edge)
    print("Incoming")
    for item in edge.incoming_edges:
        print(item)
        
    print("Outgoing")
    for item in edge.outgoing_edges:
        print(item)
    return

def sort_edges_timewise(edges,reverse):
    edges.sort(key= lambda val:val.time,reverse=reverse)
    return edges

def binary_search_find_time_greater_equal(arr, target,strictly=False):
    start = 0;
    end = len(arr) - 1;
    ans = -1;
    while (start <= end):
        mid = (start + end) // 2;
  
        # Move to right side if target is
        # greater.
        if strictly:
            if (arr[mid].time <= target):  
                start = mid + 1;
            else:
                ans = mid;
                end = mid - 1;
        else:
            if (arr[mid].time < target):
                start = mid + 1;
            else:
                ans = mid;
                end = mid - 1;
        # Move left side.
    if not strictly:  ### find the first occurrance of this target
        less_found = False
        while ans != -1 and ans > 0 and not less_found:
            if arr[ans-1].time == target:
                ans = ans - 1
            else:
                less_found= True
    return ans;

def binary_search_find_time_lesser_equal(arr, target,strictly=False):
    if arr[-1].time < target:
        return len(arr)-1
    index = binary_search_find_time_greater_equal(arr,target,strictly=False)
    #print(index)
    if index== -1:
        return index
    if strictly:
        return index-1
    else:
        
        if arr[index].time == target:
            return index
        else:
            return index - 1

# binary_search_find_time_lesser_equal(start_node_edges,44623,True)
# binary_search_find_time_greater_equal(start_node_edges,-1,True)

class Edge():
    def __init__(self,start,end,**kwargs):
        self.start = start
        self.end = end
        self.__dict__.update(kwargs)
    def __str__(self):
        s= 'start: '+str(self.start)+ " end: "+str(self.end)+ " "
        if 'time' in self.__dict__:
            s += "time: "+str(self.__dict__['time'])
        return s
class Node():
    def __init__(self,id,**kwargs):
        self.id = id
        self.__dict__.update(kwargs)
        
def prepare_alias_table_for_edge(edge,incoming = False,window_interactions = None):
    if not incoming:
        if window_interactions is None:
            window_interactions = len(edge.outgoing_edges)
        time_diffs = [item.time -edge.time for item in edge.outgoing_edges[:window_interactions]]
    else:
        if window_interactions is None:
            window_interactions = len(edge.incoming_edges)
        time_diffs = [item.time -edge.time for item in edge.incoming_edges[:window_interactions]]
    mn = np.mean(time_diffs)
    std = np.std(time_diffs)
    if len(time_diffs) == 1 or std == 0:
        std = 1
        #print(time_diffs,[-(item - mn)/std for item in time_diffs] )
    time_diffs = [-(item - mn)/std for item in time_diffs]   ### less time diff edge should be more prioritized
    time_diffs = np.exp(time_diffs)
    norm_const = sum(time_diffs)
    nbr_sample_probs = [float(prob)/norm_const for prob in time_diffs]
    J, q  = alias_setup(nbr_sample_probs)
    return nbr_sample_probs,J,q
def run_random_walk_without_temporal_constraints(edge,max_length=20,delta = 0):
    rw = []
        #max_length = random.randint(20,50)
    if len(edge.incoming_edges) > 0:  ######### Add event for 
        random_walk_start_time = edge.incoming_edges[alias_draw(edge.inJ,edge.inq)].time
    else:
        random_walk_start_time = edge.time - delta
    random_walk = [edge]
    ct = 1
    done = False
    while ct < max_length and not done:
        if len(edge.out_nbr_sample_probs) == 0:
            done = True
            random_walk.append(Edge(start=edge.end,end='end_node',time=edge.time))
        else:
            tedge = edge.outgoing_edges[alias_draw(edge.outJ,edge.outq)]
            edge = tedge
            random_walk.append(edge)
            ct += 1
    return [random_walk_start_time]+[(edge.start,edge.end,edge.time) for edge in random_walk]
    #else:
        
    
def clean_random_walk(rw):  ### essentially if next time stamp is same then it make sures not to repeat the same node again ###
    newrw = [rw[0]]
    cur_time = rw[1][2]
    cur_nodes = [rw[1][0],rw[1][1]]
    newrw.append(rw[1])
    for wk in rw[2:]:
        
        if wk[2] == cur_time:
            if wk[1] in cur_nodes:
                return newrw
            else:
                newrw.append(wk)
                cur_nodes.append(wk[1])
        else:
            newrw.append(wk)
            cur_time = wk[2]
            cur_nodes = [wk[0],wk[1]]
    return newrw
def filter_rw(rw,cut_off=6):
    if len(rw) >= cut_off:
        return True
    else:
        return False
    
    
def convert_walk_to_seq(rw):
    seq = [(rw[1][0],rw[0])]
    for item in rw[1:]:
        seq.append((item[1],item[2]))
    return seq
def convert_seq_to_id(vocab,seq):
    nseq = []
    for item in seq:
        nseq.append((vocab[item[0]],item[1]))
    return nseq
def update_delta(delta,d =.1):
    if delta == 0:
        return delta+d
    return delta
def get_time_delta(sequence,start_delta = 0):
    times = [item[1] for item in sequence]
    delta = [update_delta(a-b) for a,b in zip(times[1:],times[:-1])]
    
    delta = [update_delta(0)] + delta #+ [-1]
    return [(item[0],item[1],t) for item, t in zip(sequence,delta)]



# def get_X_Y_T_CID_from_sequences(sequences):  ### This also need to provide the cluster id of the 
#     seq_X = []
#     seq_Y = []
#     seq_Xt = []
#     seq_Yt = []
#     seq_XDelta = []
#     seq_YDelta = []
#     seq_XCID = []
#     seq_YCID = []
#     for seq in sequences:
#         seq_X.append([item[0] for item in seq[:-1]])  ## O contain node id
#         seq_Y.append([item[0] for item in seq[1:]])
#         seq_Xt.append([item[1] for item in seq[:-1]])   ## 1 contain timestamp
#         seq_Yt.append([item[1] for item in seq[1:]])
#         seq_XDelta.append([item[2] for item in seq[:-1]])   ## 2 contain delta from previous event
#         seq_YDelta.append([item[2] for item in seq[1:]])
#         seq_XCID.append([item[3] for item in seq[:-1]])   ## 3 contains the cluster id
#         seq_YCID.append([item[3] for item in seq[1:]])
#     X_lengths = [len(sentence) for sentence in seq_X]
#     Y_lengths = [len(sentence) for sentence in seq_Y]
#     max_len = max(X_lengths)
#     return seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len,seq_XCID,seq_YCID
# #seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len,seq_XCID,seq_YCID = get_X_Y_T_CID_from_sequences(sequences)

# def get_batch(start_index,batch_size,seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,seq_XCID,seq_YCID):
#     batch_X = seq_X[start_index:start_index+batch_size]
#     batch_Y = seq_Y[start_index:start_index+batch_size]
#     batch_Xt = seq_Xt[start_index:start_index+batch_size]
#     batch_Yt = seq_Yt[start_index:start_index+batch_size] 
#     batch_XDelta = seq_XDelta[start_index:start_index+batch_size]
#     batch_YDelta = seq_YDelta[start_index:start_index+batch_size]   
#     batch_X_len = X_lengths[start_index:start_index+batch_size]
#     batch_Y_len = Y_lengths[start_index:start_index+batch_size]
#     batch_XCID = seq_XCID[start_index:start_index+batch_size]
#     batch_YCID = seq_YCID[start_index:start_index+batch_size] 
#     max_len = max(batch_X_len)
#     #print(max_len)
#     pad_batch_X = np.ones((batch_size, max_len),dtype=np.int64)*pad_token
#     pad_batch_Y = np.ones((batch_size, max_len),dtype=np.int64)*pad_token
#     pad_batch_Xt = np.ones((batch_size, max_len),dtype=np.float32)*pad_token
#     pad_batch_Yt = np.ones((batch_size, max_len),dtype=np.float32)*pad_token
#     pad_batch_XDelta = np.ones((batch_size, max_len),dtype=np.float32)*pad_token
#     pad_batch_YDelta = np.ones((batch_size, max_len),dtype=np.float32)*pad_token
#     pad_batch_XCID = np.ones((batch_size, max_len),dtype=np.int64)*pad_cluster_id
#     pad_batch_YCID = np.ones((batch_size, max_len),dtype=np.int64)*pad_cluster_id
#     for i, x_len in enumerate(batch_X_len):
#         #print(i,x_len,len(batch_X[i][:x_len]),len(pad_batch_X[i, 0:x_len]))
#         pad_batch_X[i, 0:x_len] = batch_X[i][:x_len]
#         pad_batch_Y[i, 0:x_len] = batch_Y[i][:x_len]
#         pad_batch_Xt[i, 0:x_len] = batch_Xt[i][:x_len]
#         pad_batch_Yt[i, 0:x_len] = batch_Yt[i][:x_len]
#         pad_batch_XDelta[i, 0:x_len] = batch_XDelta[i][:x_len]
#         pad_batch_YDelta[i, 0:x_len] = batch_YDelta[i][:x_len]
#         pad_batch_XCID[i, 0:x_len] = batch_XCID[i][:x_len]
#         pad_batch_YCID[i, 0:x_len] = batch_YCID[i][:x_len]
#     pad_batch_X =  torch.LongTensor(pad_batch_X).to(device)
#     pad_batch_Y =  torch.LongTensor(pad_batch_Y).to(device)
#     pad_batch_Xt =  torch.Tensor(pad_batch_Xt).to(device)
#     pad_batch_Yt =  torch.Tensor(pad_batch_Yt).to(device)
#     pad_batch_XDelta =  torch.Tensor(pad_batch_XDelta).to(device)
#     pad_batch_YDelta =  torch.Tensor(pad_batch_YDelta).to(device)
#     batch_X_len = torch.LongTensor(batch_X_len).to(device)
#     batch_Y_len = torch.LongTensor(batch_Y_len).to(device)
#     pad_batch_XCID =  torch.LongTensor(pad_batch_XCID).to(device)
#     pad_batch_YCID =  torch.LongTensor(pad_batch_YCID).to(device)
#     return pad_batch_X,pad_batch_Y,pad_batch_Xt,pad_batch_Yt,pad_batch_XDelta,pad_batch_YDelta,batch_X_len,batch_Y_len,pad_batch_XCID,pad_batch_YCID

# def data_shuffle(seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,seq_XCID,seq_YCID):
#     indices = list(range(0, len(seq_X)))
#     random.shuffle(indices)
#     #### Data Shuffling
#     seq_X = [seq_X[i] for i in indices]   #### 
#     seq_Y = [seq_Y[i] for i in indices]
#     seq_Xt = [seq_Xt[i] for i in indices]
#     seq_Yt = [seq_Yt[i] for i in indices]    
#     seq_XDelta = [seq_XDelta[i] for i in indices]
#     seq_YDelta = [seq_YDelta[i] for i in indices]
#     X_lengths = [X_lengths[i] for i in indices]
#     Y_lengths = [Y_lengths[i] for i in indices]
#     seq_XCID = [seq_XCID[i] for i in indices]
#     seq_YCID = [seq_YCID[i] for i in indices]
#     return seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,seq_XCID,seq_YCID


def get_node_set_length(edges):
    nodes = set()
    for start,end,_ in edges:
        nodes.add(start)
        nodes.add(end)
    return len(nodes)