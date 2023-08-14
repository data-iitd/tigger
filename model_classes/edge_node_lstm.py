import os
import torch
import torch.nn as nn
# current_file_directory = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(current_file_directory)


class EdgeNodeLSTM(nn.Module):
    def __init__(self, vocab, node_pretrained_embedding, nb_layers, num_components, edge_attr_dim, 
                 node_attr_dim, nb_lstm_units=100, clust_dim=3, batch_size=3,device='cpu'):
        super(EdgeNodeLSTM, self).__init__()
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers  # number of LSTM layers
        self.nb_lstm_units = nb_lstm_units  # dimension of hidden layer h
        self.clust_dim = clust_dim  # dimension of the cluster embedding
        self.batch_size = batch_size
        self.edge_attr_dim = edge_attr_dim  # number of edge attributes
        self.node_attr_dim = node_attr_dim  # nomber of node attributes
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1
        self.gnn_dim = node_pretrained_embedding.shape[1]
        self.mu_hidden_dim = 100  # dimension between cluster embedding and z_gnn
        self.num_components = num_components  # number of clusters
        print("Number of components,", num_components)
        nb_vocab_words = len(self.vocab)

        # create embedding with the graphsage vector
        padding_idx = self.vocab['<PAD>']
        self.gnn_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.gnn_dim,
            padding_idx=padding_idx  # padding index it'll make the whole vector zeros
        )
        self.gnn_embedding.weight.data.copy_(torch.from_numpy(node_pretrained_embedding))
        self.gnn_embedding.weight.requires_grad = False
        
        # create cluster embedding
        self.cluster_embeddings = nn.Embedding(
            num_embeddings=self.num_components,
            embedding_dim=self.clust_dim,
            padding_idx=padding_idx
        )
        
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.clust_dim + self.gnn_dim + self.edge_attr_dim + self.node_attr_dim,   ## cluster + GNN + edge embedding
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )  
        
        # output layer which projects back to tag space
        self.embedding_hidden = nn.Linear(self.gnn_dim,self.gnn_dim)  #re-embedding gnn embedding?
        self.hidden_to_ne_hidden = nn.Linear(self.nb_lstm_units, 200)  # Z_cluster
        self.clusterid_hidden = nn.Linear(200,self.num_components)  # Z_cluster to cluster distribution
        self.cluster_mu = nn.Linear(200,self.mu_hidden_dim*self.num_components)  # mu's per cluster
        self.cluster_var = nn.Linear(200,self.mu_hidden_dim*self.num_components) # var's per cluster
        self.gnn_decoder1 = nn.Linear(self.mu_hidden_dim,400)  #layer 1 gnn_decoder
        self.gnn_decoder2 = nn.Linear(400,self.gnn_dim)  # layer 2 gnn_decoder
        self.gnn_decoder3 = nn.Linear(self.gnn_dim, self.gnn_dim)  # layer 3 gnn_decoder
        
        self.edge_decoder1 = nn.Linear(self.mu_hidden_dim,128)  #layer 1 gnn_decoder
        self.edge_decoder2 = nn.Linear(128,self.edge_attr_dim)  # layer 2 gnn_decoder
        self.edge_decoder3 = nn.Linear(self.edge_attr_dim, self.edge_attr_dim)  # layer 3 gnn_decoder
        
        self.feat_decoder1 = nn.Linear(self.mu_hidden_dim,128)  #layer 1 gnn_decoder
        self.feat_decoder2 = nn.Linear(128,self.node_attr_dim)  # layer 2 gnn_decoder
        self.feat_decoder3 = nn.Linear(self.node_attr_dim, self.node_attr_dim)  # layer 3 gnn_decoder
                
        self.mse_los_gnn = nn.MSELoss(reduction='none')
        self.mse_loss_edge = nn.MSELoss(reduction='none')
        self.mse_loss_feat = nn.MSELoss(reduction='none')
        self.celoss_cluster = nn.CrossEntropyLoss(ignore_index=0)

        self.relu_cluster = nn.LeakyReLU()  # activation forhidden to determin cluster id
        self.relu_edge = nn.LeakyReLU()  #activation for reconstruction edge attributes
        self.relu_feat = nn.LeakyReLU()
        self.relu_gnn = nn.LeakyReLU() 
        
        self.device = device
        
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale/2)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz #.sum(dim=-1)    
    
    def event_mse(self,x_hat,x):
        a =  (x-x_hat)*(x-x_hat)
        return a.sum(-1)
    
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units).to(self.device)
        hidden_b = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units).to(self.device)
        return (hidden_a, hidden_b)

    def forward(self, seq_Xvocab_id, seq_Yvocab_id, 
                seq_Xedge_attr, seq_Yedge_attr, 
                seq_Xnode_attr, seq_Ynode_attr, 
                x_length, mask, 
                kl_weight, seq_Xcluster_id, seq_Ycluster_id):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        batch_size, seq_len = seq_Xvocab_id.size()
        
        # ---------------------
        # 1. embed the input
        # --------------------
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)

        X = self.gnn_embedding(seq_Xvocab_id)  # retrieve the gnn embedding from node vocab id
        X = self.embedding_hidden(X)  #TODO WHY?
        XCID_embedding = self.cluster_embeddings(seq_Xcluster_id)  # retrieve embedding for cluster id
        X = torch.cat((seq_Xedge_attr, seq_Xnode_attr, X, XCID_embedding), -1)
        
        
        # ---------------------
        # 2. Run through RNN
        # ---------------------
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, x_length, batch_first=True,enforce_sorted=False)
        X, self.hidden = self.lstm(X, self.hidden)  # now run through LSTM
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)  # unpacking operation
        X = X.contiguous()
        
        # ---------------------
        # 3. Project to tag space
        # ---------------------
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # Predict cluster id props
        Y_hat = self.relu_cluster(self.hidden_to_ne_hidden(X))  # Z_cluster
        Y_clusterid = self.clusterid_hidden(Y_hat)  # prop distrubution over clusters.
    
        #  get the mu and variance parameters voor retrieving GNN embedding
        mu, log_var = self.cluster_mu(Y_hat), self.cluster_var(Y_hat)
        mu = mu.view(batch_size,seq_len,self.num_components,self.mu_hidden_dim)
        log_var = log_var.view((batch_size,seq_len,self.num_components,self.mu_hidden_dim))
        
        #YCID is expected to have size of batch_size*seq_len
        if self.training:  # select true cluster during training?
            Y_clusterid_sampled = seq_Xcluster_id.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)
        else:  # select argmax cluster  during inference?
            Y_clusterid_sampled = torch.argmax(Y_clusterid,dim=2)
            Y_clusterid_sampled = Y_clusterid_sampled.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)
            
        # retrieve mu and log_var for y_cluster
        mu = torch.gather(mu,2,Y_clusterid_sampled).squeeze(2)
        log_var = torch.gather(log_var,2,Y_clusterid_sampled).squeeze(2)
            
        
        std = torch.exp(log_var / 2)  # determine std for hidden layer z to gnn
        q = torch.distributions.Normal(mu, std)  # create distribution layer
        z = q.rsample()  # sample z for reconstruction of gnn embedding + edge atributes
        
        # Reconstruct gnn embedding
        ne_hat = self.gnn_decoder2(self.relu_gnn(self.gnn_decoder1(z)))  # reconstruct  GNN embeding
        ne_hat = self.gnn_decoder3(ne_hat)  #3de layer decoder gnn embedding
        
        # Reconstruct edge features
        edge_hat = self.edge_decoder2(self.relu_edge(self.edge_decoder1(z)))  # reconstruct  edge
        edge_hat = self.edge_decoder3(edge_hat)  #3de layer decoder
        
        # Reconstruct node features
        feat_hat = self.feat_decoder2(self.relu_feat(self.feat_decoder1(z)))  # reconstruct  edge
        feat_hat = self.feat_decoder3(feat_hat)  #3de layer decoder
        
        # prep y_true
        Y = self.gnn_embedding(seq_Yvocab_id)
        Y_temp = Y  #### Will be used to calculate the elbo loss
        Y = Y.view(-1,Y.shape[2])
        
        # reconstruction loss
        kl = self.kl_divergence(z, mu, std)*mask   # used for regularisation
        recon_loss_ne = self.mse_los_gnn(ne_hat,Y_temp)
        recon_loss_ne = recon_loss_ne.sum(-1)*mask
        recon_loss_edge = self.mse_loss_edge(edge_hat, seq_Yedge_attr)
        recon_loss_edge = recon_loss_edge.sum(-1)*mask
        recon_loss_feat = self.mse_loss_feat(feat_hat, seq_Ynode_attr)
        recon_loss_feat = recon_loss_feat.sum(-1)*mask
        elbo = kl_weight*kl + recon_loss_ne + recon_loss_edge + recon_loss_feat  ### recon_loss 
        num_events = mask.sum()
        elbo = elbo.sum()/num_events
        
        #cluster loss
        Y_clusterid = Y_clusterid.view(-1, Y_clusterid.shape[-1])
        loss_cluster = self.celoss_cluster(Y_clusterid, seq_Xcluster_id.view(-1))
        
        loss = elbo + loss_cluster
        
        log_dict = {
            'loss': loss.item(),
            'elbo_loss': elbo.item(),
            'kl_loss': (kl.sum()/num_events).item(),
            'reconstruction_ne': (recon_loss_ne.sum()/num_events).item(),
            'reconstruction_edge': (recon_loss_edge.sum()/num_events).item(),
            'cross_entropy_cluster': (loss_cluster.sum()/num_events).item(),
            'reconstruction_feat': (recon_loss_feat.sum()/num_events).item(),
        }      
        
        return ne_hat, edge_hat, loss, log_dict, Y_clusterid, feat_hat
