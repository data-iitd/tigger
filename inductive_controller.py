#%%
import os
import pickle
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict,Counter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy as np
import copy
import torch
import torch.optim as optim
from tgg_utils import prepare_sample_probs, Edge, Node
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from model_classes.inductive_model import get_topk_event_prediction_rate
from model_classes.edge_node_lstm import EdgeNodeLSTM
# import scann
print("loaded")

#%%
## !! vocab and node id is mixed, need to merge them
class InductiveController:
    def __init__(self, node_feature_path, edge_list_path, graphsage_embeddings_path,
                 num_epochs = 10, num_clusters = 500, window_interactions = 6,
                 n_walks=20000, l_w = 20, minimum_walk_length = 2, verbose=2,
                 pca_components=3, config_path = "temp/", lr=.001, batch_size = 1024,
                 kl_weight = 0.00001):
        self.feature_path = node_feature_path  # dataframe path with node feature attributes
        self.edge_list_path = edge_list_path  # dataframe with edge lists incl features
        self.graphsage_embeddings_path = graphsage_embeddings_path
        self.config_dir = self.prep_config_dir(config_path)
        self.gpu_num = -1
        self.config_path = "temp/"
        self.num_epochs = num_epochs
        self.num_clusters = num_clusters
        self.pca_components = pca_components  # no of pca dimensions used for clustering
        self.window_interactions = window_interactions  # ??
        self.n_walks = n_walks  # number of walks samples
        self.l_w = l_w  # maximum length of a walk
        self.minimum_walk_length = minimum_walk_length
        self.verbose = verbose
        self.lr = lr  # learning rate
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.device = self.get_device()

        #prep data
        self.data = pd.read_parquet(edge_list_path)  # edge list
        self.edge_attr_cols = [c for c in self.data.columns if c not in ['start', 'end']]
        self.edges, self.node_id_to_object = self.create_node_and_edge_objects_with_links_lists()
        self.vocab = self.create_lstm_vocab()
        self.node_features = self.create_feature_matrix_from_pandas(self.feature_path)
        
        self.node_embedding_matrix, self.normalized_dataset = self.create_node_embedding_matrix_from_dict()
        self.cluster_labels = self.reduce_embedding_dim_and_cluster()
        self.define_sample_with_prob_per_edge()
        
        #prep model
        self.model, self.optimizer = self.initialize_model()
        
        
        if verbose >=2:
            print(f"number of edges in data file {self.data.shape[0]}")
            print(f"attributes found for edges: {self.edge_attr_cols}")
            print(f"length of edges {len(self.edges)} length of nodes {len(self.node_id_to_object)}")
            print(f"Length of vocab {len(self.vocab)}")
            print(f"Id of end node {self.vocab['end_node']}")
            print(f"Id of padding node {self.vocab['<PAD>']}")
            
        # checks.
        assert self.data.shape[0]==len(self.edges), \
            "The number of edges is different then in the edge list file"
        assert len(self.vocab) - 2 == len(self.node_id_to_object), \
            "not all nodes are in the vocab"
        #TODO asset len(self.node_id_to_object)== shape[0] node features
        
        
    def create_node_and_edge_objects_with_links_lists(self):
        """creates a list of  node and edge objects with renumbered id's 
        The node object has an incoming and outgoing edge list.
        """
        edges = []
        node_id_to_object = {}
        
        for row in self.data.values:
            start = int(row[0])
            end = int(row[1])
            
            # add start and end node to node_dict
            if start not in node_id_to_object:
                node_id_to_object[start] = Node(id=start, as_start_node=[], as_end_node=[])
            if end not in node_id_to_object:
                node_id_to_object[end] = Node(id=end, as_start_node=[], as_end_node=[])
                
            #add edge to edge dict
            edge = Edge(start=start, end=end, attributes=row[2:], outgoing_edges = [],incoming_edges=[])
            edges.append(edge) 
            node_id_to_object[start].as_start_node.append(edge)  # add edge to start node list
            node_id_to_object[end].as_end_node.append(edge)      # add edge to end node list.
        return (edges, node_id_to_object)
        
    def define_sample_with_prob_per_edge(self):
        """Adds a list of connected edges to each edge object together with the 
        probability"""
        dist = []
        for edge in tqdm(self.edges): 
            end_node_edges = self.node_id_to_object[edge.end].as_start_node  #get succesive edges
            edge.outgoing_edges = end_node_edges
            edge.out_nbr_sample_probs = prepare_sample_probs(edge)  # assigns uniform probability.
            #TODO test other types of probability distributions.
            if self.verbose >=2:
                dist.append(len(end_node_edges))

        if self.verbose >=2:
            plt.hist(dist)
            plt.xscale("log")
            plt.title("histogram of edge degree distribution")
            plt.show()
        
    def create_lstm_vocab(self):
        vocab = {'<PAD>': 0,'end_node':1}
        for node in self.data['start']:
            if node not in vocab:
                vocab[node] = len(vocab)
        for node in self.data['end']:
            if node not in vocab:
                vocab[node] = len(vocab)
                  
        return vocab
        
    def sample_random_Walks(self):
        """Create n_walk number of random walks"""
        if self.verbose >= 2:
            print(f"Running Random Walk on {len(self.edges)} edges")
        random_walks = []
        for edge in tqdm(random.sample(self.edges, self.n_walks)):
            rw = self.run_random_walk(edge)
            if rw is not None:
                random_walks.append(rw)
            
        if self.verbose >= 2:
            print(f"collected {len(random_walks)} random walks")
            print("Average length of random walks")
            lengths = []
            for wk in random_walks:
                lengths.append(len(wk))
            print(f"Mean length {np.mean(lengths):.02f} and Std deviation {np.std(lengths):.02f}")
            plt.hist(lengths)
            plt.xscale("log")
            plt.title("histogram of random walk lengths")
            plt.show()

        return random_walks

    def run_random_walk(self, edge):
        """Creates a single random walk starting from an edge
        every step consists of a tupple with
        (edge attribute, vocab end node id, cluster id, node_features)
        """
        edge_shape = edge.attributes.shape
        feat_dim = self.node_features.shape[1]
        vocab_id = self.vocab[edge.end]
        random_walk = [(list(edge.attributes), vocab_id, self.cluster_labels[edge.end],
                         self.node_features.iloc[vocab_id].values.tolist())]
        done = False
        ct = 0
        while ct < self.l_w and not done: 
            if len(edge.outgoing_edges) == 0:
                done = True
                random_walk.append((list(np.zeros(edge_shape)), 1, 1, list(np.zeros(feat_dim))))  # end vocab id and cluster
            else:
                edge = np.random.choice(edge.outgoing_edges, 1, edge.out_nbr_sample_probs)[0]
                vocab_id = self.vocab[edge.end]
                random_walk.append((list(edge.attributes), vocab_id, self.cluster_labels[vocab_id], 
                                    self.node_features.iloc[vocab_id].values.tolist()))
                ct += 1
        return random_walk if len(random_walk) >= self.minimum_walk_length else None
       
    def create_node_embedding_matrix_from_dict(self):
        """create a numpy matrix of the node embedding order by vocab"""
        node_embeddings_feature = pickle.load(open(self.graphsage_embeddings_path,"rb"))
        node_emb_size = node_embeddings_feature[0].shape[0]
        
        node_embedding_matrix = np.zeros((len(self.vocab),node_emb_size))
        for item in self.vocab:
            if item == '<PAD>':
                arr = np.zeros(node_emb_size)
            elif item == 'end_node':
                arr = np.ones(node_emb_size)
            else:
                arr = node_embeddings_feature[item]
            index = self.vocab[item]
            node_embedding_matrix[index] = arr
       
        # create row normalized dataset excluding <padding>
        norm = np.linalg.norm(node_embedding_matrix, axis=1)
        norm[norm==0] = 1  # set zero to 1 to avoid dividing by zero
        normalized_dataset = node_embedding_matrix / norm[:, np.newaxis] 
        
        
        if self.verbose >= 2:
             print(f"Node embedding matrix has shape {node_embedding_matrix.shape}") 
        
        return (node_embedding_matrix, normalized_dataset)  
    
    def create_feature_matrix_from_pandas(self, parquet_filename):
        feat_df = pd.read_parquet(parquet_filename)  #has id column with node number
        vocab_df = (
            pd.DataFrame
            .from_dict(self.vocab, orient='index', columns=['vocab_id'], dtype='int64')
            .reset_index(names='id')
            .merge(feat_df, on='id', how='outer')
            .fillna(0) 
            .set_index('vocab_id')   
            .drop('id', axis=1)        
        )
        
        #check if all rows of the feature are mapped to the vocab
        assert feat_df.shape[0] + 2 == vocab_df.shape[0], \
            "feat df has {feat_df.shape[0]} rows and vocab has {vocab_df.shape[0]} instead of {feat_df.shape[0] + 2}"
            
        return vocab_df
        
    
    def reduce_embedding_dim_and_cluster(self):
        """reduced the embedding dimension with PCA and cluster the reduced embed dim
        returns a list of cluster labels ordered by vocab"""
        pca = PCA(n_components=self.pca_components)
        node_embedding_matrix_pca = pca.fit_transform(self.node_embedding_matrix[2:]) ### since 0th and 1st index is of padding and end node
        
        kmeans = (KMeans(n_clusters=self.num_clusters, random_state=0,max_iter=10000)
                  .fit(node_embedding_matrix_pca)
        )
        labels = kmeans.labels_
        cluster_labels = [0,1]+[item+2 for item in labels]   
        
        if self.verbose >= 2:
            print(f"PCA variance explained ={np.sum(pca.explained_variance_ratio_)}")
            
            label_freq = Counter(labels)
            plt.hist(label_freq.values())
            plt.xscale("log")
            plt.title("histogram of cluster frequencies")
            plt.show()
            
            print(len(cluster_labels))
            max_label = np.max(cluster_labels)
            print("Max cluster label",max_label)
        
        return cluster_labels

    def prep_config_dir(self, config_path):
        config_dir = config_path ### Change in random walks
        isdir = os.path.isdir(config_dir) 
        if not isdir:
            os.mkdir(config_dir)
        isdir = os.path.isdir(config_dir+"/models") 
        if not isdir:
            os.mkdir(config_dir+"/models")   
        return config_dir
    
    def get_device(self):
        try:
            device = torch.device("mps")
        except:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
        if self.verbose >= 2:
            print(f"Computation device {device}")
        return device
        
    def get_X_Y_from_sequences(self, sequences):  
        """
        splits the sequences into X and Y values.
        Y values are next values in the sequence
        """
        elements = {
            0: 'edge_attr',
            1: 'vocab_id',
            2: 'cluster_id',
            3: 'node_attr'
        }
        res_dict ={'x_length': []}
        for i, e in elements.items():
            res_dict['seq_X'+e] = []
            res_dict['seq_Y'+e] = []
            
        for seq in sequences:
            for i, e in elements.items():
                res_dict['seq_X'+e].append([item[i] for item in seq[:-1]])
                res_dict['seq_Y'+e].append([item[i] for item in seq[1:]])
            res_dict['x_length'].append(len(seq)-1)
           
        return res_dict

    def get_batch(self, start_index, batch_size, seqs):
        """Creates padded batch copied to the torch device
        seqs is a dict containing :seq_Xedge, seq_Yedge, seq_X, seq_Y, 
        X_lengths, Y_lengths, seq_XCID, seq_YCID, max_len"""
        # edge_attr_dim = len(seqs['seq_Xedge'][0][0])  # dimension of the edge attributes
        
        batch_seq = {}
        pad_batch_seq = {}
        pad_value = self.vocab['<PAD>']
        for k,seq in seqs.items():
            if k == 'x_length':
                x_length = seqs['x_length'][start_index:start_index+batch_size]
            else:
                batch_seq[k] = seq[start_index:start_index+batch_size]
                if type(seq[0][0])==list:
                    dim = len(seq[0][0])
                    padding_shape = (batch_size, self.l_w, dim)
                else:
                    padding_shape = (batch_size, self.l_w)
                pad_batch_seq[k] = np.ones(padding_shape, dtype=np.int32) * pad_value
               
        
        for i, x_len in enumerate(x_length):
            for k, pad_seq in pad_batch_seq.items():
                pad_seq[i, 0:x_len] = batch_seq[k][i]
                
        pad_batch_seq['x_length'] = x_length
        
        for k, pad_seq in pad_batch_seq.items():
            if k[5:] in ['vocab_id', 'cluster_id'] or k == 'x_length':
                pad_batch_seq[k] = torch.LongTensor(pad_seq).to(self.device)
            else:
                pad_batch_seq[k] = torch.FloatTensor(pad_seq).to(self.device)
        
        return pad_batch_seq
        
    def data_shuffle(self, seqs):
        #seq_Xedge, seq_Yedge, seq_X, seq_Y, X_lengths, Y_lengths, seq_XCID, seq_YCID
        indices = list(range(len(seqs['x_length'])))
        random.shuffle(indices)
        #### Data Shuffling
        for k,v in seqs.items():
            k = [v[i] for i in indices]    
        return seqs
    
    def initialize_model(self):
        edge_dim = self.edges[0].attributes.shape[0]  # edge dimension
        node_attr_dim = self.node_features.shape[1]
        elstm = EdgeNodeLSTM(
            vocab=self.vocab, 
            node_pretrained_embedding=self.normalized_dataset,
            nb_layers=2, 
            nb_lstm_units=128,
            edge_attr_dim=edge_dim,
            node_attr_dim=node_attr_dim,
            clust_dim=64, # used for cluster embedding
            batch_size=self.batch_size,
            device=self.device,
            num_components=self.num_clusters + 2  #incl padding + end cluster
        )
        elstm = elstm.to(self.device)
        
        optimizer = optim.Adam(elstm.parameters(), lr=.001)
        
        if self.verbose >= 2:
            num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)
            print(f"Number of parameters {num_params}")
        return (elstm, optimizer)
    
    def train_model(self):
        epoch_wise_loss = []
        seqs = self.sample_random_Walks()
        seqs = self.get_X_Y_from_sequences(seqs)
        
        loss_dict = {
                'loss': [],
                'elbo_loss': [],
                'reconstruction_ne': [],
                'reconstruction_edge': [],
                'reconstruction_feat': [],
                'kl_loss': [],
                'cross_entropy_cluster': []
            }
           
        for epoch in range(self.num_epochs):
            self.model.train()
            seqs = self.data_shuffle(seqs)
            n_seqs = len(seqs['x_length'])
            
            for start_index in range(0, n_seqs-self.batch_size, self.batch_size):              
                print("\r%d/%d" %(int(start_index),n_seqs),end="")
                wt_update_ct = 0
                pad_seqs = self.get_batch(start_index, self.batch_size, seqs)
                self.model.zero_grad()
                mask_distribution = (pad_seqs['seq_Yvocab_id']!=0)
                mask_distribution = mask_distribution.to(self.device)
                
                # forward + backward pas
                _, _, loss, log_dict, _, _ = self.model(
                    **pad_seqs, mask=mask_distribution, kl_weight=self.kl_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                
                for k in loss_dict.keys():
                    loss_dict[k].append(log_dict[k])
    
                wt_update_ct += 1
                if self.verbose>=2 and wt_update_ct%10 == 0:
                    for k,v in loss_dict.items():
                        print(f"{k} = {np.mean(v[-print_ct:])}")
     
            running_loss = np.mean(loss_dict['loss'][-wt_update_ct:])
            epoch_wise_loss.append(running_loss)
            
            if self.verbose>=1:
                print(f"\r\rEpoch {epoch} done \r")
                for k,v in loss_dict.items():
                        print(f"{k} = {np.mean(v[-wt_update_ct:])}")
            
            if epoch%20 == 0:
                print("Running evaluation")
                # evaluate_model(elstm)
            
        
        ### Saving the model
        state = {
            'model':self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': running_loss
        }
        torch.save(state, self.config_dir+"/models/best_model.pth".format(str(epoch)))
        
        return (epoch_wise_loss, loss_dict)



#%%
if __name__ == "__main__":
    node_feature_path = "data/bitcoin/feature_attributes.parquet"
    edge_list_path = "data/bitcoin/edgelist_with_attributes.parquet"
    graphsage_embeddings_path = "graphsage_embeddings/bitcoin/embeddings.pkl"
    n_walks=200
    inductiveController = InductiveController(
        node_feature_path=node_feature_path,
        edge_list_path=edge_list_path,
        graphsage_embeddings_path=graphsage_embeddings_path,
        n_walks=n_walks,
        batch_size = 24
    )
    seqs = inductiveController.sample_random_Walks()
    seqs = inductiveController.get_X_Y_from_sequences(seqs)
    seqs = inductiveController.data_shuffle(seqs)
    seqs = inductiveController.get_batch(0, 24, seqs)
    
    epoch_wise_loss, loss_dict = inductiveController.train_model()
    
# %%
