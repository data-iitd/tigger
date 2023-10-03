#%%
import os
import pickle
import importlib
import random
import warnings
import math
import pandas as pd
from datetime import datetime
from collections import defaultdict,Counter
from tqdm.auto import tqdm  
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
from sklearn.neighbors import BallTree
import tigger_package.edge_node_lstm
importlib.reload(tigger_package.edge_node_lstm)
from tigger_package.edge_node_lstm import EdgeNodeLSTM
try:
    import matplotlib.pyplot as plt
except:
    pass
# import scann
print("loaded")

#%%
## !! vocab and node id is mixed, need to merge them
class InductiveController:
    def __init__(self, nodes, edges, embed, path, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)
                 
        self.config_dir = path + config_dict['config_path']  
        os.makedirs(self.config_dir, exist_ok=True)
        self.model_dir = self.config_dir + "models/"
        os.makedirs(self.model_dir, exist_ok=True)
                
        self.gpu_num = -1
        self.device = self.get_device()
        random.seed(1)

        #prep data
        self.data = edges
        self.edge_attr_cols = [c for c in self.data.columns if c not in ['start', 'end']]
        self.edges, self.node_id_to_object = self.create_node_and_edge_objects_with_links_lists()
        self.vocab = self.create_lstm_vocab()
        self.node_features = self.create_feature_matrix_from_pandas(nodes)
        
        self.node_embedding_matrix, self.normalized_dataset = self.create_node_embedding_matrix_from_dict(embed)
        self.cluster_labels, self.kmeans, self.pca = self.reduce_embedding_dim_and_cluster()
        self.define_sample_with_prob_per_edge()
        
        #prep model
        self.model, self.optimizer = self.initialize_model()
    
        if self.verbose >=2:
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
        
    def sample_random_Walks(self, train=True):
        """Create n_walk number of random walks"""
        if self.verbose >= 2:
            print(f"Running Random Walk on {len(self.edges)} edges")
        random_walks = []
        n_walks = self.n_walks if train else self.test_n_walks
        for edge in tqdm(random.sample(self.edges, n_walks)):
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
            # plt.xscale("log")
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
        random_walk = [(list(edge.attributes), 
                        self.normalized_dataset[vocab_id].tolist(), 
                        self.cluster_labels[vocab_id],
                        self.node_features.iloc[vocab_id].values.tolist())]
        done = False
        ct = 0
        while ct < self.l_w and not done: 
            if len(edge.outgoing_edges) == 0:
                done = True
                random_walk.append(
                    (list(np.ones(edge_shape)),
                     self.normalized_dataset[1].tolist(),
                     self.cluster_labels[1],
                     list(np.ones(feat_dim)))
                    )  # end vocab id and cluster
            else:
                edge = np.random.choice(edge.outgoing_edges, 1, edge.out_nbr_sample_probs)[0]
                vocab_id = self.vocab[edge.end]
                random_walk.append((list(edge.attributes), 
                                    self.normalized_dataset[vocab_id].tolist(), 
                                    self.cluster_labels[vocab_id], 
                                    self.node_features.iloc[vocab_id].values.tolist()))
                ct += 1
        return random_walk if len(random_walk) >= self.minimum_walk_length else None
       
    def create_node_embedding_matrix_from_dict(self, embed):
        """create a numpy matrix of the node embedding order by vocab"""
        # node_embeddings_feature = pickle.load(open(self.graphsage_embeddings_path,"rb"))
        node_emb_size = embed.shape[1]
        
        node_embedding_matrix = np.zeros((len(self.vocab),node_emb_size))
        for item in self.vocab:
            if item == '<PAD>':
                arr = np.zeros(node_emb_size)
            elif item == 'end_node':
                arr = np.ones(node_emb_size)
            else:
                arr = embed.loc[item,:].values
            index = self.vocab[item]
            node_embedding_matrix[index] = arr
       
        # create row normalized dataset excluding <padding>
        norm = np.linalg.norm(node_embedding_matrix, ord=np.inf, axis=1)
        norm[norm==0] = 1  # set zero to 1 to avoid dividing by zero
        normalized_dataset = node_embedding_matrix / norm[:, np.newaxis] 
        
        
        if self.verbose >= 2:
             print(f"Node embedding matrix has shape {node_embedding_matrix.shape}") 
        
        return (node_embedding_matrix, normalized_dataset)  
    
    def create_feature_matrix_from_pandas(self, nodes):
        nodes  #has id as index with node number
        node_attr = (
            pd.DataFrame
            .from_dict(self.vocab, orient='index', columns=['vocab_id'], dtype='int64')
            # .reset_index(names='id')
            .merge(nodes, how='left', left_index=True, right_index=True)
            .fillna(0) 
            .set_index('vocab_id')   
            # .drop('id', axis=1)        
        )
        node_attr.iloc[self.vocab['end_node']]=1
        
        #check if all rows of the feature are mapped to the vocab
        if nodes.shape[0] + 2 != node_attr.shape[0]:
            warnings.warn("feat df has {feat_df.shape[0]} rows and vocab has {vocab_df.shape[0]} instead of {feat_df.shape[0] + 2}. This can be cause by unconnected nodes.")
            
        return node_attr
        
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
        
        return cluster_labels, kmeans, pca

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
            1: 'node_embed',
            2: 'cluster_id',
            3: 'node_attr'
        }
        res_dict = defaultdict(list)            
        for seq in sequences:
            for i, e in elements.items():
                res_dict[e].append([item[i] for item in seq])
            res_dict['x_length'].append(len(seq))
           
        return res_dict

    def get_batch(self, start_index, batch_size, seqs):
        """Creates padded batch copied to the torch device
        seqs is a dict containing :
        seq_edge, seq_X, X_lengths, seq_CID"""
        
        batch_seq = {}
        pad_batch_seq = {}
        pad_value = self.vocab['<PAD>']
        # copy relevant part of seq into batch dic and create padding matrices
        for k,seq in seqs.items():
            if k == 'x_length':
                x_length = seqs['x_length'][start_index:start_index+batch_size]
            else:
                batch_seq[k] = seq[start_index:start_index+batch_size]
                if type(seq[0][0])==list:
                    dim = len(seq[0][0])
                    padding_shape = (batch_size, self.l_w+1, dim)
                else:
                    padding_shape = (batch_size, self.l_w+1)
                pad_batch_seq[k] = np.ones(padding_shape, dtype=np.float64) * pad_value
               
        
        # join padding matrix with batch sequences
        for i, x_len in enumerate(x_length):
            for k, pad_seq in pad_batch_seq.items():
                pad_seq[i, 0:x_len] = batch_seq[k][i]
                
        pad_batch_seq['x_length'] = [l-1 for l in x_length]
        
        # convert to tensor
        x_batch = {}
        y_batch = {}
        for k, pad_seq in pad_batch_seq.items():               
            if k in ['cluster_id', 'x_length']:
                torch_seq = torch.LongTensor(pad_seq).to(self.device)
            else:
                torch_seq = torch.FloatTensor(pad_seq).to(self.device)
                
            # split in x and y
            if k == 'x_length':
                x_batch[k] = torch_seq.to('cpu')
                y_batch[k] = torch_seq.to('cpu')
            else:
                x_batch[k] = torch_seq[:, :-1] 
                y_batch[k] = torch_seq[:, 1:] 
                
        x_batch.pop('edge_attr')
   
        return x_batch, y_batch
        
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
        embed_dim = self.normalized_dataset.shape[1]
        elstm = EdgeNodeLSTM(
            vocab=self.vocab, 
            gnn_dim=embed_dim,
            nb_layers=self.nb_lstm_layers, 
            nb_lstm_units=128,
            edge_attr_dim=edge_dim,
            node_attr_dim=node_attr_dim,
            clust_dim=64, # used for cluster embedding
            batch_size=self.batch_size,
            device=self.device,
            kl_weight=self.kl_weight,
            num_components=self.num_clusters + 2,  #incl padding + end cluster
            dropout = self.dropout
        )
        elstm = elstm.to(self.device)
        
        optimizer = optim.Adam(elstm.parameters(), lr=.001, weight_decay=self.weight_decay)
        
        if self.verbose >= 2:
            num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)
            print(f"Number of parameters {num_params}")
        return (elstm, optimizer)
    
    def train_model(self):
        epoch_wise_loss = []
        running_loss = 0 
        val_loss_epoch = 0
        seqs = self.sample_random_Walks()
        seqs = self.get_X_Y_from_sequences(seqs)
        
        test_seqs = self.sample_random_Walks(train=False)
        test_seqs = self.get_X_Y_from_sequences(test_seqs)
        
        loss_dict = {
                'loss': [],
                'elbo_loss': [],
                'reconstruction_ne': [],
                'reconstruction_edge': [],
                'reconstruction_feat': [],
                'kl_loss': [],
                'cross_entropy_cluster': []
            }
        val_loss = []
        val_dict_list = []
           
        for epoch in range(self.num_epochs):
            self.model.train()
            seqs = self.data_shuffle(seqs)  # shuffle data
            n_seqs = len(seqs['x_length'])  # number of walks
            
            for start_index in range(0, n_seqs-self.batch_size+1, self.batch_size):              
                batch_cnt = 0  # Batch number in Epoch
                x_batch, y_batch = self.get_batch(start_index, self.batch_size, seqs)
                self.model.zero_grad()
                
                # forward + backward pas
                y_hat= self.model(**x_batch)
                loss, log_dict = self.model.train_los(**y_hat, **y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                
                for k in loss_dict.keys():
                    loss_dict[k].append(log_dict[k])
    
                batch_cnt += 1
                
                print(f"\r {int(start_index)} / {n_seqs}, epoch:{epoch} loss={running_loss}, val_loss: {val_loss_epoch}",end="")
     
            running_loss = np.mean(loss_dict['loss'][-batch_cnt:])
            epoch_wise_loss.append(running_loss)
            
            if self.verbose>=3:
                print(f"\r\rEpoch {epoch} done \r")
                for k,v in loss_dict.items():
                        print(f"{k} = {np.mean(v[-batch_cnt:])}")
            
            if epoch%5 == 0:
                val_loss_epoch, val_dict = self.evaluate_model(test_seqs)
                val_dict_list.append(val_dict)
                val_loss.append(val_loss_epoch)
            
            print(f"\r {int(start_index)} / {n_seqs}, epoch:{epoch} loss={running_loss}, val_loss: {val_loss_epoch}",end="")
            
        
        ### Saving the model
        state = {
            'model':self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': running_loss
        }
        torch.save(state, self.model_dir+"/best_model.pth".format(str(epoch)))
        loss_dict['epoch_loss'] = epoch_wise_loss
        loss_dict['val_loss'] = val_loss
        loss_dict['val_dict'] = self.mean_dict(val_dict_list, calc_mean=False)
        
        if self.verbose>=1:
            self.plot_loss(loss_dict)
        
        return (loss_dict)
    
    def evaluate_model(self, test_seqs):
        """calculates the test loss over the complete epoch"""
        self.model.eval()
        epoch_wise_loss = []
        val_log_dicts = []
        n_seqs = len(test_seqs['x_length'])  # number of walks
        for start_index in range(0, n_seqs-self.batch_size+1, self.batch_size):
            print("\r%d/%d" %(int(start_index),n_seqs),end="")
            batch_cnt = 0  # Batch number in Epoch
            x_batch, y_batch = self.get_batch(start_index, self.batch_size, test_seqs)
                
            # forward + backward pas
            y_hat= self.model(**x_batch)
            _, log_dict = self.model.train_los(**y_hat, **y_batch)
                
            batch_cnt += 1
            epoch_wise_loss.append(log_dict['loss'])
            val_log_dicts.append(log_dict)
            
            
            
        return (np.mean(epoch_wise_loss), self.mean_dict(val_log_dicts))
              
    def mean_dict(self, dict_list, calc_mean=True):
        res = {}
        
        for k in dict_list[0].keys():
            res[k] = [d[k] for d in dict_list]
            if calc_mean:
                res[k] = np.mean(res[k])
                
        return res
            
    
    def lin_grid_search(self, grid_dict):
        grid_param = list(grid_dict.keys())[0]
        vals = grid_dict[grid_param]
        res = {}
        
        for val in vals:
            setattr(self, grid_param, val)
            self.model, self.optimizer = self.initialize_model()
            loss_dict = self.train_model()
            run = {
                'grid_param': grid_param,
                'val': val,
            }
            res[val]={**run, **loss_dict}
            
        if self.verbose>=1:
            self.plot_grid(res)
        return res
    
    def plot_grid(self, res):
        losses = []
        val_losses = []
        for k, v in res.items():
            losses.append(v['epoch_loss'][-1])
            val_losses.append(v['val_loss'][-1])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        keys = [str(k) for k in res.keys()]
        ind = np.arange(len(keys))
        width = 0.2
        ax1.bar(ind-width, losses, 2*width, label='loss')
        ax1.bar(ind+width, val_losses, 2*width, label='val_loss')
        ax1.set_xticklabels(keys)
        ax1.legend()
        for k, v in res.items():
            ax2.plot(v['val_loss'], label=str(k))
        ax2.legend()
        print(f"loss: {losses}")
        print(f"val loss: {val_losses}")
        plt.show()
        
    def plot_loss(self, loss_dict):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(loss_dict['epoch_loss'], label='loss')
        ax1.plot(loss_dict['val_loss'], label='val_loss')
        ax1.legend()
        
        val_losses = loss_dict['val_dict']
        for k,v in val_losses.items():
            ax2.plot(v, label=k)
        ax2.legend(bbox_to_anchor=(1.5, 1.))
        ax2.set_yscale("log")
        plt.show()
    
    def embed_to_cluster(self, embed):
        embed_reduced = self.pca.transform(embed)
        cluster = self.kmeans.predict(embed_reduced)
        return cluster
    
    def synthetic_nodes_to_seqs(self, node_ids, nodes):
        """convert list consisting of node embed, node attr and edge attr concatenated
        into seqs dict"""
        embed_dim = self.normalized_dataset.shape[1]
        
        seqs = {}
        seqs['node_id'] = node_ids
        seqs['has_ended'] = []  # used to track sequences for which an end node is generated
        seqs['node_embed'] = [[list(s)] for s in nodes.iloc[node_ids, :embed_dim].values]
        seqs['cluster_id'] = [[s] for s in self.embed_to_cluster(nodes.iloc[node_ids, :embed_dim].values)]
        seqs['node_attr'] = [[list(s)] for s in nodes.iloc[node_ids, embed_dim:].values]
        
        for k, pad_seq in seqs.items():               
            if k in ['cluster_id']:
                seqs[k] = torch.LongTensor(pad_seq).to(self.device)
            if k in ['node_embed', 'node_attr']:
                seqs[k] = torch.FloatTensor(pad_seq).to(self.device)
        return seqs
    
    def merge_inference_step(self, generated_seq_batch, y_hat):
        for k, seq in generated_seq_batch.items():
            generated_seq_batch[k] = torch.cat((seq,y_hat[k]), 1)
            
    def merge_inference_batch(self, generated_seqs, generated_seqs_batch):
        for k, seq in generated_seqs_batch.items():
            generated_seqs[k] = generated_seqs.get(k, []) + seq.tolist()
            
    def extract_edge(self, x_batch, new_x_batch, end_node_id):
        edges = []
        has_ended = x_batch['has_ended']
        for i, end_id in enumerate(new_x_batch['node_id']):
            if i not in has_ended:  # no end node has been generated earlier
                if end_id != end_node_id:
                    start_id = x_batch['node_id'][i]
                    edge_attr = new_x_batch['edge_attr'][i][0].tolist()
                    edges.append((start_id, end_id, edge_attr))
                else:
                    has_ended.append(i)
        new_x_batch['has_ended'] = has_ended
        return edges
            
    def y_hat_to_x_batch(self, y_hat, searcher, synthetic_nodes):
        """removes the _hat phrase from the dict keys and maps the inferece node embed and edge
        to the generated synthetic nodes"""
        embed_dim = self.normalized_dataset.shape[1]
        x_batch = {}
        
        # remove -hat phrase in key
        for k,v in y_hat.items():
            if k != 'cluster_id_hat_vector':
                x_batch[k[:-4]] = v
                
        # get nearest synthic node
        inferred_node_vector = torch.cat((x_batch['node_embed'], x_batch['node_attr']), -1)
        _, mapped_id = searcher.query(torch.squeeze(inferred_node_vector, 1).tolist(), k=1)
        mapped_synth_node = synthetic_nodes.iloc[np.squeeze(mapped_id, 1)]
        x_batch['node_id'] = np.squeeze(mapped_id, 1).tolist()
        
        
        # replace inference node prop with mapped synthetic node
        x_batch['node_embed'] = [[list(s)] for s in mapped_synth_node.iloc[:, :embed_dim].values]
        x_batch['node_embed'] = torch.FloatTensor(x_batch['node_embed']).to(self.device)
        x_batch['cluster_id'] = [[s] for s in self.embed_to_cluster(mapped_synth_node.iloc[:, :embed_dim].values)]
        x_batch['cluster_id'] = torch.LongTensor(x_batch['cluster_id']).to(self.device)
        x_batch['node_attr'] = [[list(s)] for s in mapped_synth_node.iloc[:, embed_dim:].values]
        x_batch['node_attr'] = torch.FloatTensor(x_batch['node_attr']).to(self.device)
        
        return x_batch
        
    def get_input_batch(self, x_batch):
        input_batch = {}
        for k in ['node_embed', 'cluster_id', 'node_attr']:
            input_batch[k] = x_batch[k]
        
        input_batch['x_length'] = [1]*x_batch['node_embed'].shape[0]
        return input_batch

    def add_end_node(self, synthetic_nodes):
        embed_dim = self.normalized_dataset.shape[1]
        node_attr_dim = self.node_features.shape[1]
        end_node_id = synthetic_nodes.shape[0]
        synthetic_nodes.loc[end_node_id] = [1]*(embed_dim+node_attr_dim)
        return end_node_id
        
    def remove_end_nodes(self, new_x_batch, end_node_id):
        x_batch = {}
        no_end_node_ids = []
        for i, node_id in enumerate(new_x_batch['node_id']):
            if node_id != end_node_id:
                no_end_node_ids.append(i)
           
        id_tensor = torch.LongTensor(no_end_node_ids).to(self.device)     
        for k,v in new_x_batch.items():
            if k != 'node_id':
                x_batch[k] = torch.index_select(v, 0, id_tensor)
            else:
                x_batch[k] = [v[i] for i in no_end_node_ids]
            
        return x_batch
    
    def create_synthetic_walks(self, synthetic_nodes, target_cnt):
        """create walks using the synthetics nodes as starting point"""
        
        no_batches = math.ceil(target_cnt / self.batch_size)
        
        self.model.eval()
        node_count = synthetic_nodes.shape[0]
        end_node_id = self.add_end_node(synthetic_nodes)
        searcher = BallTree(synthetic_nodes, leaf_size=40)  # used to map the walks to the synth nodes
        
        
        generated_seqs = []
        for i in range(no_batches): 
            node_ids = random.choices(range(node_count), k=self.batch_size)
            x_batch = self.synthetic_nodes_to_seqs(node_ids, synthetic_nodes)
            self.model.init_hidden()  # set the hidden state of lstm to zero 
            step = 0
            
            while step < self.l_w and len(x_batch['has_ended']) < self.batch_size:
                y_hat = self.model(**self.get_input_batch(x_batch))
                new_x_batch = self.y_hat_to_x_batch(y_hat, searcher, synthetic_nodes)
                generated_seqs = generated_seqs + self.extract_edge(x_batch, new_x_batch, end_node_id)
                x_batch = new_x_batch
                step += 1
        
        return generated_seqs
                
                
        

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
