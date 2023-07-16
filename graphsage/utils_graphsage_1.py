import networkx as nx
import scipy.sparse as sp
import numpy as np
import igraph
import powerlaw
import statistics
import math
from scipy.sparse.csgraph import connected_components
from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.autograd import grad as torch_grad
from statsmodels.distributions.empirical_distribution import ECDF

import torch
import torch.nn as nn
import os
import random
import time
import heapq
import functools

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges(adj, test_frac=.1,train_neg_ratio=10, val_frac=.05, prevent_disconnect=True, verbose=False,test_neg_ratio=10):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_array(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj) # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu) # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0] # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac)) # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac)) # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples) # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < test_neg_ratio*num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false:
            continue
            
        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < train_neg_ratio*len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, 
            # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
            false_edge in test_edges_false or \
            false_edge in val_edges_false or \
            false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
        val_edges, val_edges_false, test_edges, test_edges_false


#Embedding and Link prediction part
def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def evaluate_overlap_torch(_N,_num_of_edges,adj_origin,embedding_matrix_numpy,link_prediction_from_embedding_one_to_other):
    
    import heapq
    predict_adj=np.zeros((_N,_N)).astype(int)
    h=[]
    num_h=0
    for i in range(_N):
        print('\r%d/%d'%(i,_N),end="")
        nowsarr=link_prediction_from_embedding_one_to_other(i,embedding_matrix_numpy)
        nowsarr=nowsarr.detach().to('cpu').numpy()
        for j in range(i+1,_N):
            nows=nowsarr[j]
            if num_h<_num_of_edges:
                heapq.heappush(h,(nows,i,j))
                num_h=num_h+1
            else:
                if h[0][0]<nows:
                    heapq.heappop(h)
                    heapq.heappush(h,(nows,i,j))
    for x in h:
        a=x[1]
        b=x[2]
        predict_adj[a][b]=1
        predict_adj[b][a]=1
    print(h[:10])
    maxh=0
    for x in h:
        if x[0]>maxh:
            maxh=x[0]
    print(maxh)
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(_N):
        for j in range(_N):
          if predict_adj[i,j]==1 and adj_origin[i,j]==1:
              tp=tp+1
          if predict_adj[i,j]==0 and adj_origin[i,j]==1:
              fp=fp+1
          if predict_adj[i,j]==1 and adj_origin[i,j]==0:
              fn=fn+1
          if predict_adj[i,j]==0 and adj_origin[i,j]==0:
              tn=tn+1
    print(predict_adj.shape)
    print(np.sum(predict_adj))
    print(np.sum(adj_origin))
    total_num=_N*_N
    print('True Positve:%d, %.2f'%(tp,tp/(tp+fp)))
    print('False Positve:%d, %.2f'%(fp,fp/(tp+fp)))
    print('True Negative:%d, %.2f'%(tn,tn/(tn+fn)))
    print('False Negative:%d, %.2f'%(fn,fn/(tn+fn)))
    print('Positive:%.2f'%((tp+fp)/total_num))
    print('Negative:%.2f'%((tn+fn)/total_num))
    return predict_adj




def symmetric(directed_adjacency, clip_to_one=True):
    """
    Symmetrize the input adjacency matrix.
    Parameters
    ----------
    directed_adjacency: sparse matrix or np.array of shape (N,N)
                        Input adjacency matrix.
    clip_to_one: bool, default: True
                 Whether the output should be binarized (i.e. clipped to 1)

    Returns
    -------
    A_symmetric: sparse matrix or np.array of the same shape as the input
                 Symmetrized adjacency matrix.

    """

    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric

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
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees),1)).power_law.alpha


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

def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()

def compute_graph_statistics(A_in):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
          
    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """

    A = A_in.copy()

    assert ((A == A.T).all())
    A_graph = nx.from_numpy_matrix(A).to_undirected()

    statistics = {}

    d_max, d_min, d_mean = statistics_degrees(A)

    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean

    # largest connected component
    LCC = statistics_LCC(A)

    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)

    # claw count
    statistics['claw_count'] = statistics_claw_count(A)

    # triangle count
    statistics['triangle_count'] = statistics_triangle_count(A)

    # Square count
    statistics['square_count'] = statistics_square_count(A)

    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)

    # gini coefficient
    statistics['gini'] = statistics_gini(A)

    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)

    # Clustering coefficient
    statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / statistics['claw_count']

    # Number of connected components
    statistics['n_components'] = connected_components(A)[0]
      
    statistics['cpl'] = statistics_compute_cpl(A)

    return statistics

class SupervisedGraphSage(nn.Module):

    def __init__(self,enc,_N, device):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        embed_dim=self.enc.embed_dim
        self.device = device
        self.xent= nn.BCELoss().to(self.device)
        
        self.fc1 = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fc2 = nn.Linear(embed_dim,1).to(self.device)
        #self.fc3 = nn.Linear(embed_dim,embed_dim).cuda()
        #self.fc4 = nn.Linear(embed_dim,embed_dim).cuda()
        self._N=_N
    
    def forward(self,edges_list):
        node1=edges_list[:,0]
        node2=edges_list[:,1]
        x=self.enc(node1).to(self.device)
        x=torch.t(x)
        y=self.enc(node2).to(self.device)
        y=torch.t(y)
        
        out = x*y
        out = self.fc1(out)
        out = F.leaky_relu(out,0.2)
        out = self.fc2(out)
        out = torch.sigmoid(out).squeeze()
        return out

    def loss(self, edges_list, labels):
        scores=self.forward(edges_list)
        return self.xent(scores, labels.squeeze())
    
    def train_acc(self,x,y):
        #Apply softmax to output. 
        pred=self.forward(x)
        true=torch.from_numpy(y).int().to(self.device).reshape(-1)
        # true=torch.from_numpy(y).int().reshape(-1)
        pred=(pred>0.5).int()
        ans=torch.sum((pred==true).int()).item()
        return ans/len(pred)
    
    def validation_step(self, x_val, y_val, batch_size=1024):
        batches = int(math.floor(y_val.shape[0]/batch_size))
        val_loss = []
        for i in range(batches):
            x_batch = x_val[range(i*batch_size, (i+1)*batch_size)]
            y_batch = y_val[range(i*batch_size, (i+1)*batch_size)]
            loss = self.loss(x_batch, torch.FloatTensor(y_batch).to(self.device))
            val_loss.append(loss)
            
        return statistics.fmean(val_loss)
        
        
        
    def train(self,train,labels,epochs,optimizer, x_val = None, y_val = None, batch_size = 1024):
        
        train_loss = []
        val_loss = []
        epoch_id = []
        val_epoch = []
        for epoch in range(epochs):
            batch=random.sample(range(len(train)),batch_size)
            start_time = time.time()
            batch_edges = train[batch]
            batch_labels=labels[batch]
            optimizer.zero_grad()
            loss = self.loss(batch_edges,Variable(torch.FloatTensor(batch_labels)).to(self.device))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            if epoch % 10==0:
                print('\rEpoch:%d,Loss:%f,estimated time:%.2f'%(epoch, loss.item(),(end_time-start_time)*(epochs-epoch)),end="")
                train_loss.append(loss.item())
                epoch_id.append(epoch)
            if epoch%100==0 and x_val is not None and y_val is not None:
                val_loss.append(self.validation_step(x_val, y_val))
                val_epoch.append(epoch)
                # print('\n acc:'+str(self.train_acc(train,labels)))
        return {'train_loss': train_loss, 'epoch': epoch_id, 'val_loss': val_loss, 'val_epoch': val_epoch}

class GraphSAGE:
    
    def __init__(self, _N, adj_dic, feat_matrix, embedding_dim, level=1, verbose_level=4):
        self._N =_N
        self._M = 0  # this is set during training
        self.adj_dic=adj_dic  # node dict of neighbor list
        self.verbose = verbose_level  # 1) no output, 2) minimum output, 3)performance, 4) debug
        self.num_feat=feat_matrix.shape[1]  ### node type 0/ 1
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device
        
        if self.verbose> 1:
            print("number of features", self.num_feat)
            print("number of nodes,", self._N)
            print(f"device is {self.device}")
            
        
        self.features = nn.Embedding(_N,self.num_feat)
        self.features.weight = nn.Parameter(torch.FloatTensor(feat_matrix), requires_grad=False)
        self.features.to(self.device)
        self.embedding_dim=embedding_dim
        
        self.agg1 = MeanAggregator(self.features, device=self.device)
        self.enc1 = Encoder(self.features, self.num_feat, self.embedding_dim, self.adj_dic, self.agg1, gcn=False, device=self.device)
        self.agg2 = MeanAggregator(lambda nodes : self.enc1(nodes).t())
        self.enc2 = Encoder(lambda nodes : self.enc1(nodes).t(), self.enc1.embed_dim, self.embedding_dim, self.adj_dic, self.agg2,base_model=self.enc1, gcn=False, device=self.device)
        if level == 1:
            self.graphsage = SupervisedGraphSage(self.enc1,self._N, device=self.device)
        else:
            self.graphsage = SupervisedGraphSage(self.enc2,self._N, device=self.device)
        embedding_matrix_torch=torch.t(self.graphsage.enc(range(self._N)))
        self.embedding_matrix_numpy=embedding_matrix_torch.detach().to('cpu').numpy()
        
    
    def save_model(self,path='graph_graphsage.pth',embedding_path='embeddings'):
        print("Saving to , ", path , embedding_path)
        torch.save(self.graphsage.state_dict(), path)
        
        embedding_matrix_torch=torch.t(self.graphsage.enc(range(self._N)))
        self.embedding_matrix_numpy=embedding_matrix_torch.detach().to('cpu').numpy()
        
        np.save(embedding_path,self.embedding_matrix_numpy)
        
    def load_model(self,path='graph_graphsage.pth',embedding_path='embeddings.npy'):
        self.graphsage.load_state_dict(torch.load(path), strict=False)
        self.embedding_matrix_numpy = np.load(embedding_path).reshape((self._N,self.embedding_dim))
        
        
    def graphsage_link_prediction_from_embedding_one_to_other(self,i,embedding):
        I_list=[]
        J_list=[]
        for idx in range(self._N):
            I_list.append(i)
            J_list.append(idx)
        node1=torch.Tensor(embedding[I_list,:].astype(float)).to(self.device)
        x=Variable(node1)
        node2=torch.Tensor(embedding[J_list,:].astype(float)).to(self.device)
        y=Variable(node2)
        
        out = x*y
        out = self.graphsage.fc1(out)
        out = F.leaky_relu(out,0.2)
        out = self.graphsage.fc2(out)
        out = torch.sigmoid(out).squeeze()
        return out

    def get_embeddings(self):
        embedding_matrix_torch=torch.t(self.graphsage.enc(range(self._N)))
        self.embedding_matrix_numpy=embedding_matrix_torch.detach().to('cpu').numpy()
        return self.embedding_matrix_numpy

    def graphsage_train(self,boost_times=20,add_edges=1000,training_epoch=10000,
                        boost_epoch=5000,learning_rate=0.001,save_number=0,dirs='graphsage_model/'):
        
        train_stats = []  # list to keeptrack of the training results
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        
        # create train, validation and test split together with negative examples.
        datasets = self.get_train_validation_test_set(0.1, 0.0)
        datasets_false, exempt_set = self.get_false_edges_datasets(1, 1, 10, datasets)
          
        train_dataset = np.concatenate([datasets['train_set'], datasets_false['train_set']])
        labels_dataset = np.concatenate([np.ones(len(datasets['train_set'])), np.zeros(len(datasets_false['train_set']))])
        val_dataset = np.concatenate([datasets['validation_set'], datasets_false['validation_set']])
        val_label = np.concatenate([np.ones(len(datasets['validation_set'])), np.zeros(len(datasets_false['validation_set']))])
        
        # train graphsage model
        optimizer = torch.optim.Adam(self.graphsage.parameters(), lr=learning_rate,weight_decay=1e-5)
        train_metrics = self.graphsage.train(train_dataset,labels_dataset,training_epoch,optimizer, x_val=val_dataset, y_val=val_label)
        train_metrics['label'] = 'train'
        train_stats.append(train_metrics)
        self.save_model(path=dirs+'/graphsage'+str(save_number)+'.pth',
                        embedding_path=dirs+'/embedding_matrix'+str(save_number)+'.pth')
        if self.verbose >= 4:
            evaluate_overlap_torch(_N=self._N,
                                _num_of_edges=self._M,
                                adj_origin=self.adj_origin,
                                embedding_matrix_numpy=self.embedding_matrix_numpy,
                                link_prediction_from_embedding_one_to_other=self.graphsage_link_prediction_from_embedding_one_to_other)
        
        print('\n Start boosting')
        for boost_iter in range(boost_times):
            print('boost iter:%d'%(boost_iter ))
            
            cnt=0
            boost_find_iter=0
            boost_max_find_iter = 10
            while(cnt<add_edges and boost_find_iter<boost_max_find_iter):
                boost_find_iter+=1
                if self.verbose >=3:
                    print('\rtrain_add:%d'%(cnt),end="")
                add_train, _ = self.create_false_edges(10000, exempt_set)  # create new edges
                add_train_np = np.array(add_train)
                
                add_preds = self.graphsage.forward(add_train_np) # get predictions
                add_preds = add_preds.detach().cpu().numpy()
                add_train_np = add_train_np[add_preds>0.5]  # select additional training with positive predictions
                
                # add additional examples to the training set
                max_size = min(add_edges-cnt, add_train_np.shape[0])  # number of data points to be added
                if max_size > 0:
                    train_dataset =np.vstack((train_dataset,add_train_np[:max_size]))
                    labels_dataset=np.concatenate((labels_dataset, np.zeros(max_size)), axis=None)
                    
                    cnt = cnt + max_size  # update counter for added training examples
                    exempt_set = exempt_set.union(set(add_train))   # update the exampt list

            if self.verbose >= 3:
                print('\ntrain added: '+str(cnt))
                print('current training set length: ' + str(len(train_dataset)))
                print('current save path: ' + dirs+ '/graphsage'+str(save_number)+'_'+str(boost_iter)+'.pth')
            
            # optimizer = torch.optim.Adam(self.graphsage.parameters(), lr=learning_rate,weight_decay=1e-5)
            boost_metrics = self.graphsage.train(train_dataset, labels_dataset, boost_epoch, optimizer, x_val=val_dataset, y_val=val_label)
            boost_metrics['label'] = 'boost_'+str(boost_iter)
            train_stats.append(boost_metrics)
            self.save_model(path=dirs+'/graphsage'+str(save_number)+'_'+str(boost_iter)+'.pth',
                            embedding_path=dirs+'/embedding_matrix'+str(save_number)+'_'+str(boost_iter)+'.pth')
            if self.verbose >= 4:
                evaluate_overlap_torch(_N=self._N,
                                    _num_of_edges=self._M,
                                    adj_origin=self.adj_origin,
                                    embedding_matrix_numpy=self.embedding_matrix_numpy,
                                    link_prediction_from_embedding_one_to_other=self.graphsage_link_prediction_from_embedding_one_to_other)
        
        return train_stats
    
    def join_metrics_dics(self, train_metrics, boost_metrics):
        for k, v in train_metrics.items():
            train_metrics[k] = v + boost_metrics[k]
        return train_metrics
            
    def get_train_validation_test_set(self, validation_frac, test_frac):
        """creates a train, validation and test set from the node dict containing the neighbors 
        using the specific fractions"""
        
        #create edge list
        edges = []
        for src, neighbors in self.adj_dic.items():
            for dst in neighbors:
                if dst > src: # avoid duplicates and self loop because edges are in the dict in both directions
                    edges.append((src, dst))
                    
        self._M = len(edges)
        num_test = int(np.floor(self._M * test_frac)) # size of test set
        num_val = int(np.floor(self._M * validation_frac)) # size of train set
        assert (num_test + num_val) < (self._N * 0.8), "validation + test is greater 80% of total dataset"
        
              
        # create train, validation and test set
        random.shuffle(edges)
        test_set = edges[:num_test]
        validation_set = edges[num_test: num_test+num_val]
        train_set = edges[num_test+num_val:]
        
        assert (len(test_set)+len(validation_set)+len(train_set) == self._M),  "length of datasets is unequal to the number of edges"  
        return {"train_set": train_set, "validation_set": validation_set, "test_set": test_set}
    
    def create_false_edges(self, size, exempt_set):
        """creates edges that are not in the edge list and neither in hte exempt set"""
        false_edges = []
        
        while len(false_edges) < size:
            src = np.random.randint(0, self._N)
            dst = np.random.randint(0, self._N)
            if src == dst:
                continue
            
            false_edge = (min(src, dst), max(src, dst))

            #ensure that edge is not in exempt_set
            if false_edge in exempt_set:
                continue

            false_edges.append(false_edge)
            exempt_set.add(false_edge)
        return false_edges, exempt_set
        
    def get_false_edges_datasets(self, ratio_train, ration_val, ratio_test, datasets):
        """ create a train, val and test dataset containing false edges"""
        # create exempt dataset ass union of all datasets.
        exempt_set = functools.reduce(lambda a,b: set(a).union(set(b)), datasets.values())
        
        false_datasets = {}
        for ratio, key in zip([ratio_train, ration_val, ratio_test], datasets.keys()):
            size = len(datasets[key]) * ratio  # determine size of the false dataset
            false_datasets[key], exempt_set = self.create_false_edges(size, exempt_set)  #create and add false dataset
            
        return false_datasets, exempt_set
            
    
            
#GAN part
            
def generate_ecdf(embeddings):
    hist=sklearn.metrics.pairwise_distances(X=embeddings, metric='euclidean').reshape(-1,1)
    print(hist.shape)
    _ = plt.hist(hist, bins=30)  # arguments are passed to np.histogram
    plt.title("Histogram")
    plt.show()
    ecdf_embedding = ECDF(hist.reshape(-1,))
    return ecdf_embedding



class Generator(nn.Module):
    def __init__(self,noise_dim,embedding_dim, g_hidden_dim=[],
                 batch_size=16
                 ):
        super(Generator, self).__init__()
        
        self.g_hidden_dim=g_hidden_dim
        self.W_up=nn.Linear(in_features=self.g_hidden_dim[-1], out_features=embedding_dim, bias=True)
        self.batch_size=batch_size
        ances=noise_dim
        self.fc1=nn.Linear(in_features=noise_dim, out_features=g_hidden_dim[0])
        self.fc2=nn.Linear(in_features=g_hidden_dim[0], out_features=g_hidden_dim[1])
        if len(self.g_hidden_dim)==3:
            self.fc3=nn.Linear(in_features=g_hidden_dim[1], out_features=g_hidden_dim[2])
    
    
    # forward method
    def forward(self, x):
        
        temp=x
        temp=self.fc1(temp)
        temp=F.leaky_relu(temp,0.3)
        temp=self.fc2(temp)
        temp=F.leaky_relu(temp,0.3)
        if len(self.g_hidden_dim)==3:
            temp=self.fc3(temp)
            temp=F.leaky_relu(temp,0.3)
        
        temp=self.W_up(temp)
        temp=F.tanh(temp)
        
        return temp
    
class Discriminator(nn.Module):
    def __init__(self,embedding_dim, d_hidden_dim=[],
                 batch_size=16):
        
        super(Discriminator, self).__init__()
        self.len_hidden_dim=len(d_hidden_dim)
        
        self.W_down=nn.Linear(in_features=embedding_dim, out_features=d_hidden_dim[0], bias=True)
        if self.len_hidden_dim==3:
            self.fc1=nn.Linear(in_features=d_hidden_dim[0], out_features=d_hidden_dim[1])
            self.fc2=nn.Linear(in_features=d_hidden_dim[1], out_features=d_hidden_dim[2])
            self.fc3=nn.Linear(in_features=d_hidden_dim[2], out_features=1)
        elif self.len_hidden_dim==2:
            self.fc1=nn.Linear(in_features=d_hidden_dim[0], out_features=d_hidden_dim[1])
            self.fc2=nn.Linear(in_features=d_hidden_dim[1], out_features=1)
        else:
            self.fc1=nn.Linear(in_features=d_hidden_dim[0], out_features=1)
            
    # forward method
    def forward(self, x):
        temp=self.W_down(x)
        temp=F.leaky_relu(temp,0.3)
        temp=self.fc1(temp)
        if self.len_hidden_dim>=2:
            temp=F.leaky_relu(temp,0.3)
            temp=self.fc2(temp)
        if self.len_hidden_dim>=3:
            temp=F.leaky_relu(temp,0.3)
            temp=self.fc3(temp)
        temp=F.tanh(temp)
        temp=torch.mean(temp)
        return temp.view(1)
    

def sample_real_data(data,batch_size):
    idx=random.sample(range(data.shape[0]),batch_size)
    return data[idx,:]


def calc_gradient_penalty(netD, real_data, fake_data,batch_size):
    
    #print ("real_data: ", real_data.size(), fake_data.size(),batch_size)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(self.device)
    #print(alpha.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(self.device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    #print("interpolate size,",disc_interpolates.size())
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #print("Gradient size ",gradients.size())
    gradients = gradients.view(gradients.size(0), -1)
    #print(gradients.size(), " after norm ", gradients.norm(2, dim=1).size())
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def eval_plot(netG,embedding_matrix,noise_dim,mmd_beta=1):
    hist_real=sklearn.metrics.pairwise_distances(X=embedding_matrix, metric='euclidean').reshape(-1,)
    ecdf_embedding_matrix = ECDF(hist_real)
    plt.plot(ecdf_embedding_matrix.x,ecdf_embedding_matrix.y, label="original embedding")
    
    noise=torch.randn(embedding_matrix.shape[0],noise_dim).to(self.device)
    sample=netG(noise).detach().cpu().numpy()
    hist_fake=sklearn.metrics.pairwise_distances(X=sample, metric='euclidean').reshape(-1,)
    ecdf_generate = ECDF(hist_fake)
    plt.plot(ecdf_generate.x,ecdf_generate.y, label="GAN generated")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    mmd=calculate_mmd(sample,embedding_matrix,beta=mmd_beta)
    return mmd,np.sum(hist_fake<1e-4)
    
def calculate_mmd(x1, x2, beta):
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()

    print("MMD means", x1x1.mean(),x1x2.mean(),x2x2.mean())

    return diff

def gaussian_kernel(x1, x2, beta = 1.0):
    L=sklearn.metrics.pairwise_distances(x1,x2).reshape(-1)
    return np.exp(-beta*np.square(L))
    
def gan_train(embedding_matrix_numpy,batch_size=256,noise_dim=16,g_hidden_dim=[16,32,48],d_hidden_dim=[48,16],
             lendataloader=200,Diter=5,Giter=1,epoch_numbers=10000,eval_epoch=100,save_idx=0,learning_rate=1e-4,
             mmd_beta=1,mmd_criterion=0.01,mmd_best_criterion=0.001,most_training_epoch_number=20000, dirs = 'gan_model/'):
    

    if not os.path.exists(dirs):
        os.makedirs(dirs)
            
    embedding_dim=embedding_matrix_numpy.shape[1]
    netG = Generator(noise_dim,embedding_dim, g_hidden_dim,batch_size)
    netD = Discriminator(embedding_dim, d_hidden_dim,batch_size)
    
    netD = netD.cuda()
    netG = netG.cuda()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.9),weight_decay=1e-6)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.9),weight_decay=1e-6)

    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.cuda()
    mone = mone.cuda()
    
    clamp_lower, clamp_upper = -0.01,0.01
    gen_iterations = 0
    inputv = torch.FloatTensor(batch_size, embedding_dim).cuda()
    noise = torch.randn(batch_size, noise_dim).cuda()
    hisD=[]
    hisG=[]
    h_mean=[]
    save_number=0
    epoch=0
    accelerate=False
    best_mmd=1000
    
    while(epoch<epoch_numbers):
        epoch+=1
        
        i = 0
        while i < lendataloader:
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = Diter
            j = 0
            while j < Diters and i < lendataloader:
                j += 1
    
            # clamp parameters to a cube
    #             for p in netD.parameters():
    #                 p.data.clamp_(clamp_lower, clamp_upper)
    
                data = sample_real_data(embedding_matrix_numpy,batch_size)
                i += 1

                # train with real
                real_cpu = torch.tensor(data).float()
                netD.zero_grad()
                #batch_size = real_cpu.size(0)
    
                real_cpu = real_cpu.cuda()
                inputv.resize_as_(real_cpu).copy_(real_cpu)
                inputv1 = Variable(inputv)
                errD_real = netD(inputv1)
                errD_real.backward(one)

                # train with fake
                noise = torch.randn(batch_size, noise_dim).cuda()
                #noise=random_generator(batch_size,noise_dim)
                with torch.no_grad():
                    noisev = Variable(noise) # totally freeze netG
                    fake = Variable(netG(noisev).data)
                inputv2 = fake
                errD_fake = netD(inputv2)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                hisD.append(errD_real)
                
                # train with gradient penalty
                gradient_penalty = -calc_gradient_penalty(netD, inputv1, inputv2,batch_size)
                gradient_penalty.backward()
            # print "gradien_penalty: ", gradient_penalty

    #             D_cost = D_real - D_fake + gradient_penalty
    #             Wasserstein_D = D_real - D_fake
                optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            for j in range(Giter):
                netG.zero_grad()
                # in case our last batch was the tail batch of the dataloader,
                # make sure we feed a full batch of noise
                noise = torch.randn(batch_size, noise_dim).cuda()
                #noise=random_generator(batch_size,noise_dim)
                noisev = Variable(noise)
                fake = netG(noisev)
                errG = netD(fake)
                errG.backward(one)
                optimizerG.step()
                gen_iterations += 1
            hisG.append(errG)
        if epoch%10==0:
            print("\rEpoch:%d/%d"%(epoch,epoch_numbers),end="")
        if epoch>0 and epoch%eval_epoch==0:
            #torch.save(netG.state_dict(), 'gan_model/netG'+str(save_idx)+'_'+str(int(save_number/10))+str(int(save_number%10))+'.pth')
            #torch.save(netD.state_dict(), 'gan_model/netD'+str(save_idx)+'_'+str(int(save_number/10))+str(int(save_number%10))+'.pth')
            mmd,histfakenumber=eval_plot(netG,embedding_matrix_numpy,noise_dim,mmd_beta=mmd_beta)
            #print('save:',save_number)
            print('mmd=%f,collapse=%f'%(mmd,histfakenumber/(embedding_matrix_numpy.shape[0]*embedding_matrix_numpy.shape[0])))
            save_number+=1
            if mmd<mmd_best_criterion:
                print("Best criteria matched, saving it")
                torch.save(netG.state_dict(), dirs+'/bestG.pth')
                torch.save(netD.state_dict(), dirs+'/bestD.pth')
                return True,mmd
            if mmd<mmd_criterion and accelerate==False:
                torch.save(netG.state_dict(), dirs+'/bestG.pth')
                torch.save(netD.state_dict(), dirs+'/bestD.pth')
                epoch_numbers=most_training_epoch_number
                accelerate=True
            if accelerate==True and mmd<best_mmd:
                torch.save(netG.state_dict(), dirs+'/bestG.pth')
                torch.save(netD.state_dict(), dirs+'/bestD.pth')
                best_mmd=mmd
            if histfakenumber>=embedding_matrix_numpy.shape[0]*embedding_matrix_numpy.shape[0]/2:
                return False,best_mmd
    if accelerate==True:
        return False,best_mmd
    
    return False,best_mmd
                        
            
def evaluate_overlap_torch_generate(_N,_num_of_edges,probability_matrix_generate):
    
    predict_adj=np.zeros((_N,_N))
    h=[]
    num_h=0
    for i in range(_N):
        for j in range(i+1,_N):
            nows=max(probability_matrix_generate[i][j],probability_matrix_generate[j][i])
            if num_h<_num_of_edges:
                heapq.heappush(h,(nows,i,j))
                num_h=num_h+1
            else:
                if h[0][0]<nows:
                    heapq.heappop(h)
                    heapq.heappush(h,(nows,i,j))
        print("\r%d/%d"%(i,_N),end="")
                
    for x in h:
        a=x[1]
        b=x[2]
        predict_adj[a][b]=1
        predict_adj[b][a]=1
    print(h[:10])
    
    maxh=0
    minh=1
    for x in h:
        if x[0]>maxh:
            maxh=x[0]
        if x[0]<minh:
            minh=x[0]
    print(' max: '+str(max(h))+' min: '+str(min(h)))
    
    graphic_seq_generate=[0 for i in range(_N)]
    
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(_N):
        for j in range(_N):
            if predict_adj[i,j]==1:
                graphic_seq_generate[i]+=1
    return predict_adj,graphic_seq_generate

def generate_probability_matrix(_N,embeddings,link_prediction_from_embedding_one_to_other):
    probability_matrix_generate=np.zeros((_N,_N))
    for i in range(_N):
        prob_i=link_prediction_from_embedding_one_to_other(i,embeddings).detach().to('cpu').numpy()
        for j in range(i+1,_N):
            probability_matrix_generate[i][j]=prob_i[j]
            probability_matrix_generate[j][i]=prob_i[j]
        print("\r%d/%d"%(i,_N),end="")
    return probability_matrix_generate
    
def revised_Havel_Hakimmi_Algorithm(_N,_num_of_edges,dic,probability_matrix_generate,graphic_seq_generate):
    graphic_seq=[0 for i in range(_N)]
    for i in range(_N):
        graphic_seq[i]=len(dic[i])
    graphic_seq.sort()
    print(len(graphic_seq))
    print(graphic_seq[:10])
    print(graphic_seq[:-11:-1])

    allocate_graphic_seq=[]
    for i in range(_N):
        allocate_graphic_seq.append((graphic_seq_generate[i],i))
    allocate_graphic_seq.sort()
    print(allocate_graphic_seq[:10])
    print(allocate_graphic_seq[:-11:-1])
    
    degree_of_generate=[[0,0] for i in range(_N)]
    for i in range(_N):
        x=allocate_graphic_seq[i][1]
        degree_of_generate[x][0]=graphic_seq[i]
        degree_of_generate[x][1]=x
    degree_of_generate.sort()
    degree_of_generate=degree_of_generate[::-1]
    print(degree_of_generate[:10])
    print(degree_of_generate[:-11:-1])
    
    adj_list_generate={}
    for i in range(_N):
        adj_list_generate[i]=[]
    
    remain_edge=_num_of_edges
    while(1):
        location_in_generate={}
        for i in range(_N):
            temp=degree_of_generate[i][1]
            location_in_generate[temp]=i
    
    
        x=degree_of_generate[0][1]
        adj_number=degree_of_generate[0][0]
        prob_xadj_arr=[]
        
        for i in range(_N):
            if not i==x:
                nowprob=max(probability_matrix_generate[i][x],probability_matrix_generate[x][i])
                prob_xadj_arr.append((nowprob,i))
        
        prob_xadj_arr.sort()
        prob_xadj_arr=prob_xadj_arr[::-1]
        i=0
        add_edge=False
        while(i<_N-1):
            y=prob_xadj_arr[i][1]
            locaty=location_in_generate[y]
            
            if adj_number<=0:
                break
            if degree_of_generate[locaty][0]>0 and (y not in adj_list_generate[x]):
                adj_list_generate[x].append(y)
                adj_list_generate[y].append(x)
                adj_number-=1
                remain_edge=remain_edge-1
                add_edge=True
            i=i+1
        degree_of_generate=[[0,0] for i in range(_N)]
        for i in range(_N):
            temp=allocate_graphic_seq[i][1]
            degree_of_generate[temp][0]=graphic_seq[i]-len(adj_list_generate[temp])
            degree_of_generate[temp][1]=temp
        degree_of_generate.sort()
        degree_of_generate=degree_of_generate[::-1]
        print('\r remain_edge:%d,x=%d'%(remain_edge,x),end="")
        if add_edge==False:
            break
        if remain_edge<=0:
            break
    adj_graphic_sq_generate=np.zeros((_N,_N))
    for i in range(_N):
        for j in adj_list_generate[i]:
            adj_graphic_sq_generate[i][j]=1
            
    print(np.sum(adj_graphic_sq_generate))
    sum_diag=0
    for i in range(_N):
        sum_diag+=adj_graphic_sq_generate[i][i]
    print(sum_diag)
    
    sum_symm=0
    for i in range(_N):
        for j in range(i+1,_N):
            if adj_graphic_sq_generate[i][j]==1 and adj_graphic_sq_generate[i][j]==adj_graphic_sq_generate[j][i]:
                sum_symm+=1
    print(sum_symm)
    return adj_graphic_sq_generate
    