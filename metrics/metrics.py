import os
import sys
import path
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
import sklearn
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components
import powerlaw
import time
import argparse
import igraph
from numba import jit
directory = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(directory)
sys.path.append(parent)
from tgg_utils import Edge
#print(Edge)
from networkx.algorithms.centrality import closeness_centrality,betweenness_centrality

def calculate_mmd(x1, x2, beta):
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()

    #print("MMD means", x1x1.mean(),x1x2.mean(),x2x2.mean())

    return diff

def gaussian_kernel(x1, x2, beta = 1.0):
    L=pairwise_distances(x1,x2).reshape(-1)
    return np.exp(-beta*np.square(L))

def average_metric(method_metric, repeated, header, i):
    for metric in header:
        method_metric[i][metric] = method_metric[i][metric] / repeated

def sum_metric(aaa, method_metric, i):
    header = aaa.keys()
    if len(method_metric) <= i:
        method_metric.append(aaa)
    else:
        for metric in header:
            method_metric[i][metric] = method_metric[i][metric] + aaa[metric]


def mean_median(org_graph, generated_graph, f, name):
    org_graph = np.array(org_graph)
    generated_graph = np.array(generated_graph)
    metric = np.divide(np.abs(org_graph - generated_graph), np.abs(org_graph))
    mean = np.mean(metric)
    median = np.median(metric)
    f.write('{}:\n'.format(name))
    f.write('Mean = {}\n'.format(mean))
    f.write('Median = {}\n'.format(median))
    return mean, median


def sampling(network, temporal_graph, n, p=0.5):
    for i in range(n):
        for j in range(n):
            if network[i, j] == 1 and np.random.uniform(low=0.0, high=1) <= p:
                temporal_graph[i, j] = 1

def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha

def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_matrix(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def statistics_square_count(A_in):
    """
    Compute the square count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Square count
    """

    A_igraph = igraph.Graph.Adjacency((A_in > 0).tolist()).as_undirected()
    return int(np.sum(squares(A_igraph)) / 4)

def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)

def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))
    return H_er

def statistics_cluster_props(A, Z_obs):
    def get_blocks(A_in, Z_obs, normalize=True):
        block = Z_obs.T.dot(A_in.dot(Z_obs))
        counts = np.sum(Z_obs, axis=0)
        blocks_outer = counts[:, None].dot(counts[None, :])
        if normalize:
            blocks_outer = np.multiply(block, 1 / blocks_outer)
        return blocks_outer

    in_blocks = get_blocks(A, Z_obs)
    diag_mean = np.multiply(in_blocks, np.eye(in_blocks.shape[0])).mean()
    offdiag_mean = np.multiply(in_blocks, 1 - np.eye(in_blocks.shape[0])).mean()
    return diag_mean, offdiag_mean

def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()

def compute_graph_statistics(A_in, Z_obs=None):
    A = A_in.copy()
    A_graph = nx.from_numpy_matrix(A).to_undirected()
    statistics = {}
    start_time = time.time()
    d_max, d_min, d_mean = statistics_degrees(A)
    statistics['d_mean'] = d_mean
    LCC = statistics_LCC(A)
    statistics['LCC'] = LCC.shape[0]
    statistics['wedge_count'] = statistics_wedge_count(A)
    claw_count = statistics_claw_count(A)
    statistics['power_law_exp'] = statistics_power_law_alpha(A)
    statistics['triangle_count'] = statistics_triangle_count(A)
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)
    statistics['n_components'] = connected_components(A, directed=False)[0]
    cc = closeness_centrality(A_graph)
    statistics['closeness_centrality_mean'] = np.mean(list(cc.values()))
    statistics['closeness_centrality_median'] = np.median(list(cc.values()))
    cc = betweenness_centrality(A_graph)
    statistics['betweenness_centrality_mean'] = np.mean(list(cc.values()))
    statistics['betweenness_centrality_median'] = np.median(list(cc.values()))
    if claw_count != 0:
        statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / claw_count
    else:
        statistics['clustering_coefficient'] = 0
    #print("--- %s seconds to compute connected_components ---" % (time.time() - start_time))
    if Z_obs is not None:
        # inter- and intra-community density
        intra, inter = statistics_cluster_props(A, Z_obs)
        statistics['intra_community_density'] = intra
        statistics['inter_community_density'] = inter
    #print("--- %s seconds to compute statistics_cluster_props ---" % (time.time() - start_time))
    return statistics


def calculate_between_centrality(A):
    A_graph = nx.from_numpy_matrix(A).to_undirected()
    return np.array(list(betweenness_centrality(A_graph).values()))
def calculate_closeness_centrality(A):
    A_graph = nx.from_numpy_matrix(A).to_undirected()
    return np.array(list(closeness_centrality(A_graph).values()))




def create_timestamp_edges(graphs,timestamps):   ### Club the graph snapshot with its timestamp
    edge_lists = []
    node_set = set()
    for graph, time in zip(graphs,timestamps):
        edge_list = []
        for start,adjl in graph.items():
            for end,ct in adjl.items():
                edge_list.append((start,end,time))
                node_set.add(end)
            node_set.add(start)
        edge_lists.append(edge_list)
    return edge_lists,node_set



def update_dict(dict_,key,val):
    if key not in dict_:
        dict_[key] = [val]
    else:
        dict_[key].append(val)
    return dict_
def calculate_temporal_katz_index(graphs,node_set,decay_f,No,Beta):  ## edges is a list of timestamped interactions ### Assume the edges are directional in nature
    graphs = graphs.copy()
    node_incoming_edges_dict = {}  ### it will keep only which have occured before time t (strictly less)
    katz_index = {}
    
    for index,gt in enumerate(graphs):
        print("\r%d/%d"%(index,len(graphs)),end="")
        current_node_incoming_edges_dict = {}
        for edge in gt:
            edge = Edge(start=edge[0],end=edge[1],t=edge[2],w=0)  ## vu edge
            v = edge.start
            u = edge.end
            rv = 0
            if v in node_incoming_edges_dict:
                for vedge in node_incoming_edges_dict[v]:
                    tpp = decay_f(edge.t - vedge.t,No,Beta)
                    rv += vedge.w*decay_f(edge.t - vedge.t,No,Beta)

            edge.w = rv+1
            current_node_incoming_edges_dict = update_dict(current_node_incoming_edges_dict,u,edge)
            
            if edge.t not in katz_index:
                katz_index[edge.t] = {}
            if u not in katz_index[edge.t]:
                katz_index[edge.t][u] = 0
            katz_index[edge.t][u] += (rv+1)*decay_f(0,No,Beta)
        ### Completed with one graph ###
        for node,lst in current_node_incoming_edges_dict.items():
            if node not in node_incoming_edges_dict:
                node_incoming_edges_dict[node] = []
            node_incoming_edges_dict[node] += lst
        #for node in node_set
    print()
    return katz_index



def calculate_mmds(A_O,A_S):
    statistics = {}
    a = np.expand_dims(calculate_between_centrality(A_O),1)
    b = np.expand_dims(calculate_between_centrality(A_S),1)
    statistics['mmd_betweenness_centrality'] = calculate_mmd(a,b,1)
    a = np.expand_dims(calculate_closeness_centrality(A_O),1)
    b = np.expand_dims(calculate_closeness_centrality(A_S),1)
    statistics['mmd_closeness_centrality'] = calculate_mmd(a,b,1)    
    
    return statistics

def calculate_mmd_distance(A_O,A_S):
    a = np.expand_dims(A_O,1)
    b = np.expand_dims(A_S,1)
    return calculate_mmd(a,b,1)
# nums = []
# for i in range(0,100):
#     nums.append(time_decay(i,.1,.1))
# plt.plot(nums)