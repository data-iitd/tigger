import os
import sys
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
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

from torch.distributions import Categorical
import torch.distributions as D
current_file_directory = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(current_file_directory)
from .normal import Normal
from .mixture_same_family import MixtureSameFamily
from .transformed_distribution import TransformedDistribution
from .tpp_utils import clamp_preserve_gradients

class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, 
        std_log_inter_time: Std of log-inter-event-times, 
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()

#import dpp
import torch
import torch.nn as nn
from torch.distributions import Categorical


class LogNormMix(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self, context_size= 100,mean_log_inter_time: float = 0.0,std_log_inter_time: float = 1.0,
        num_mix_components = 32
        ):
        super().__init__()
        self.context_size = context_size
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.num_mix_components = num_mix_components
        self.linear = nn.Linear(self.context_size, 3 * self.num_mix_components)




    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.linear(context)  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )




class Time2Vec(nn.Module):
    def __init__(self, activation, time_emb_size):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.f = torch.sin
        elif activation == "cos":
            self.f = torch.cos
        self.time_emb_size = time_emb_size
        self.tau_to_emb_aperiodic = nn.Linear(1,1)
        self.tau_to_emb_periodic = nn.Linear(1,self.time_emb_size-1)
        self.W = nn.parameter.Parameter(torch.randn(self.time_emb_size))
        #self.fc1 = nn.Linear(hiddem_dim, 2)
    
    def forward(self, tau):
        #x = x.unsqueeze(1)
        ## tau shape will be batch_size* seq_len
        batch_size,seq_len = tau.size()
        #tau = tau.view(-1,1)
        tau = tau.unsqueeze(-1)
        tau_ap =  self.tau_to_emb_aperiodic(tau)       ## batch_size*seq_len*time_emb
        tau_p  =  self.f(self.tau_to_emb_periodic(tau))
        #tau_p = torch.sin(self.tau_to_emb_periodic(tau)) + torch.cos(self.tau_to_emb_periodic(tau))
        tau = torch.cat([tau_ap,tau_p],axis=-1)
        tau = tau*self.W
        return tau

    
class EventLSTM(nn.Module):
    def __init__(self, vocab,node_pretrained_embedding,nb_layers,mean_log_inter_time,std_log_inter_time,num_components, nb_lstm_units=100, embedding_dim=3, 
                 batch_size=3,time_emb_dim=10,device='cpu'):
        super(EventLSTM, self).__init__()
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.time_emb_dim = time_emb_dim
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1
        self.ne_dim =self.embedding_dim #node_pretrained_embedding.shape[1]
        self.mu_hidden_dim = 100
        self.num_components =num_components
        print("Number of components,", num_components)
        # when the model is bidirectional we double the output dimension
        # self.lstm
        # build embedding layer first
        nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab['<PAD>']
        self.word_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )
        self.word_embedding.weight.data.copy_(torch.from_numpy(node_pretrained_embedding))
        self.word_embedding.weight.requires_grad = False
        
        self.time_to_embedding = nn.Linear(1,self.time_emb_dim)
        self.t2v = Time2Vec('sin',self.time_emb_dim)
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim+self.time_emb_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )  #dropout
        # output layer which projects back to tag space
        self.embedding_hidden = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.hidden_to_ne_hidden = nn.Linear(self.nb_lstm_units, 200)
        self.clusterid_hidden = nn.Linear(200,self.num_components)
        self.ne_mu = nn.Linear(200,self.mu_hidden_dim*self.num_components)
        self.ne_var = nn.Linear(200,self.mu_hidden_dim*self.num_components)
        self.ne_decoder = nn.Linear(self.mu_hidden_dim,400)
        self.ne_decoder1 = nn.Linear(400,self.ne_dim)
        self.decoder_mu = nn.Linear(self.ne_dim, self.ne_dim)
        self.decoder_std = nn.Linear(self.ne_dim,self.ne_dim)
        self.ne_log_scale = torch.zeros(1,requires_grad=True).to(device)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.hidden_to_hidden_time = nn.Linear(self.nb_lstm_units+self.embedding_dim,100)
        self.hidden_to_time = nn.Linear(100,1)
        self.sigmactivation = nn.Sigmoid()
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.relu = nn.ReLU()
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.device = device
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.lognormalmix = LogNormMix(100,self.mean_log_inter_time,self.std_log_inter_time)
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

    def forward(self, X,Y,Xt,Yt,XDelta,YDelta, X_lengths,mask,epoch,kl_weight,XCID,YCID):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        batch_size, seq_len = X.size()
        
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)

        X = self.word_embedding(X) ### need to add MLP layer after it
        Y = self.word_embedding(Y)
        #print(X.shape,Y.shape)
        #Xt = Xt.view(-1,1)
        #Xt = self.time_to_embedding(Xt)
        #Xt = Xt.view(batch_size,seq_len,Xt.shape[-1])
        X = self.embedding_hidden(X)
        Xt = self.t2v(Xt)
        X = torch.cat((X, Xt), -1)
        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        #print(X.shape,X_lengths.shape)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True,enforce_sorted=False)

        # now run through LSTM
        #print(X.shape)
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        #print(X.shape)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        #print(X.shape)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        Y_temp = Y  #### Will be used to calculate the elbo loss
        Y = Y.view(-1,Y.shape[2])
        #print(X.shape,Y.shape)
        
        # run through actual Event linear layer
        #print(X.shape)
        Y_hat = self.hidden_to_ne_hidden(X)
        Y_hat = self.relu1(Y_hat) ### Introducing non-linearity
        Y_hat = Y_hat.view(batch_size, seq_len,Y_hat.shape[-1])
        Y_clusterid = self.clusterid_hidden(Y_hat)
        
        # encode output node vector to get the mu and variance parameters
        mu, log_var = self.ne_mu(Y_hat), self.ne_var(Y_hat)
        mu = mu.view(batch_size,seq_len,self.num_components,self.mu_hidden_dim)
        log_var = log_var.view((batch_size,seq_len,self.num_components,self.mu_hidden_dim))
        #YCID is expected to have size of batch_size*seq_len
        if self.training:
            YCID = YCID.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)
            mu = torch.gather(mu,2,YCID).squeeze(2)
            log_var = torch.gather(log_var,2,YCID).squeeze(2)
        else:
            #print("In evaluation mode")
            Y_clusterid = torch.argmax(Y_clusterid,dim=2)
            Y_clusterid = Y_clusterid.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)
            mu = torch.gather(mu,2,Y_clusterid).squeeze(2)
            log_var = torch.gather(log_var,2,Y_clusterid).squeeze(2)
            
         #x = torch.arange(0,2*4*3*5)
        # x = x.view(2, 4, 3,5)
        # idx = torch.tensor([[0, 1, 2, 0],[2, 2, 2, 0]])
        # idx = idx.unsqueeze(-1).repeat(1,1,5).unsqueeze(2)
        # print(x.shape,idx.shape)

        #xt = torch.gather(x,2,idx).squeeze(2)   
        ### mu and log_var has size of batch_size,seq_len,K*100
        #log_var = clamp_preserve_gradients(log_var, -10.0, 1.0)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        kl = self.kl_divergence(z, mu, std)*mask
        # decoded
        ne_hat = self.ne_decoder1(self.relu2(self.ne_decoder(z)))
        decoder_mu = self.decoder_mu(ne_hat)
        decoder_std = self.decoder_std(ne_hat)
        ne_hat =decoder_mu
        # reconstruction loss
        #recon_loss = self.gaussian_likelihood(ne_hat, self.ne_log_scale, Y_temp)
        #recon_loss =self.gaussian_likelihood(decoder_mu, decoder_std, Y_temp)
        #self.ne_log_scale = clamp_preserve_gradients(self.ne_log_scale, -3.0, 3.0)
        #recon_loss = self.event_mse(ne_hat,Y_temp)
        recon_loss = self.mse_loss(ne_hat,Y_temp)
        recon_loss_tpp = recon_loss
        recon_loss = recon_loss.sum(-1)*mask
        elbo = kl_weight*kl + recon_loss  ### recon_loss 
        num_events = mask.sum()

        elbo = elbo.sum()/num_events
        log_dict = {
            'elbo': elbo.item(),
            'kl': (kl.sum()/num_events).item(),
            'reconstruction': (recon_loss.sum()/num_events).item(),
        }

        #Y = self.embedding_hidden(Y)
        X = torch.cat((X,ne_hat.view(-1,ne_hat.shape[2])),-1)

        #X = torch.cat((X,Y),-1)
        #print(X.shape)
        X = self.sigmactivation(X)
        X = self.hidden_to_hidden_time(X)
        X = self.sigmactivation(X)
        X = X.view(batch_size,seq_len,X.shape[-1])  
        self.inter_time_dist = self.lognormalmix.get_inter_time_dist(X) #### X is context 
        YDelta = YDelta.clamp(1e-10)
        inter_time_log_loss = self.inter_time_dist.log_prob(YDelta)   ### batch_size*seq_len
        
        return ne_hat,inter_time_log_loss,elbo,log_dict,Y_clusterid ##,log_dict,std,mu,z,recon_loss_tpp,decoder_mu,decoder_std

class EventClusterLSTM(nn.Module):
    def __init__(self, vocab,node_pretrained_embedding,nb_layers,mean_log_inter_time,std_log_inter_time,num_components, nb_lstm_units=100, embedding_dim=3, 
                 batch_size=3,time_emb_dim=10,device='cpu'):
        super(EventClusterLSTM, self).__init__()
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.time_emb_dim = time_emb_dim
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1
        self.ne_dim =self.embedding_dim #node_pretrained_embedding.shape[1]
        self.mu_hidden_dim = 100
        self.num_components =num_components
        print("Number of components,", num_components)
        # when the model is bidirectional we double the output dimension
        # self.lstm
        # build embedding layer first
        nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab['<PAD>']
        self.word_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )
        self.cluster_embeddings = nn.Embedding(
            num_embeddings=self.num_components,
            embedding_dim=self.embedding_dim - 64,
            padding_idx=padding_idx
        )
        self.word_embedding.weight.data.copy_(torch.from_numpy(node_pretrained_embedding))
        self.word_embedding.weight.requires_grad = False
        
        self.time_to_embedding = nn.Linear(1,self.time_emb_dim)
        self.t2v = Time2Vec('sin',self.time_emb_dim)
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=2*self.embedding_dim - 64+self.time_emb_dim,   ## cluster embedding+ self embedding+time embedding
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )  #dropout
        # output layer which projects back to tag space
        self.embedding_hidden = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.hidden_to_ne_hidden = nn.Linear(self.nb_lstm_units, 200)
        self.clusterid_hidden = nn.Linear(200,self.num_components)
        self.ne_mu = nn.Linear(200,self.mu_hidden_dim*self.num_components)
        self.ne_var = nn.Linear(200,self.mu_hidden_dim*self.num_components)
        self.ne_decoder = nn.Linear(self.mu_hidden_dim,400)
        self.ne_decoder1 = nn.Linear(400,self.ne_dim)
        self.decoder_mu = nn.Linear(self.ne_dim, self.ne_dim)
        self.decoder_std = nn.Linear(self.ne_dim,self.ne_dim)
        self.ne_log_scale = torch.zeros(1,requires_grad=True).to(device)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.hidden_to_hidden_time = nn.Linear(self.nb_lstm_units+self.embedding_dim,100)
        self.hidden_to_time = nn.Linear(100,1)
        self.sigmactivation = nn.Sigmoid()
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.relu = nn.ReLU()
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.device = device
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.lognormalmix = LogNormMix(100,self.mean_log_inter_time,self.std_log_inter_time)
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

    def forward(self, X,Y,Xt,Yt,XDelta,YDelta, X_lengths,mask,epoch,kl_weight,XCID,YCID):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        batch_size, seq_len = X.size()
        
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)

        X = self.word_embedding(X) ### need to add MLP layer after it
        Y = self.word_embedding(Y)
        
        XCID_embedding = self.cluster_embeddings(XCID)
        #print(X.shape,Y.shape)
        #Xt = Xt.view(-1,1)
        #Xt = self.time_to_embedding(Xt)
        #Xt = Xt.view(batch_size,seq_len,Xt.shape[-1])
        X = self.embedding_hidden(X)
        Xt = self.t2v(Xt)
        X = torch.cat((X, Xt,XCID_embedding), -1)
        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        #print(X.shape,X_lengths.shape)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True,enforce_sorted=False)

        # now run through LSTM
        #print(X.shape)
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        #print(X.shape)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        #print(X.shape)
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        Y_temp = Y  #### Will be used to calculate the elbo loss
        Y = Y.view(-1,Y.shape[2])
        #print(X.shape,Y.shape)
        
        # run through actual Event linear layer
        #print(X.shape)
        Y_hat = self.hidden_to_ne_hidden(X)
        Y_hat = self.relu1(Y_hat) ### Introducing non-linearity
        Y_hat = Y_hat.view(batch_size, seq_len,Y_hat.shape[-1])
        Y_clusterid = self.clusterid_hidden(Y_hat)
        
        # encode output node vector to get the mu and variance parameters
        mu, log_var = self.ne_mu(Y_hat), self.ne_var(Y_hat)
        mu = mu.view(batch_size,seq_len,self.num_components,self.mu_hidden_dim)
        log_var = log_var.view((batch_size,seq_len,self.num_components,self.mu_hidden_dim))
        #YCID is expected to have size of batch_size*seq_len
        if self.training:
            YCID = YCID.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)
            mu = torch.gather(mu,2,YCID).squeeze(2)
            log_var = torch.gather(log_var,2,YCID).squeeze(2)
        else:
            #print("In evaluation mode")
            Y_clusterid = torch.argmax(Y_clusterid,dim=2)
            Y_clusterid = Y_clusterid.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)
            mu = torch.gather(mu,2,Y_clusterid).squeeze(2)
            log_var = torch.gather(log_var,2,Y_clusterid).squeeze(2)
            
         #x = torch.arange(0,2*4*3*5)
        # x = x.view(2, 4, 3,5)
        # idx = torch.tensor([[0, 1, 2, 0],[2, 2, 2, 0]])
        # idx = idx.unsqueeze(-1).repeat(1,1,5).unsqueeze(2)
        # print(x.shape,idx.shape)

        #xt = torch.gather(x,2,idx).squeeze(2)   
        ### mu and log_var has size of batch_size,seq_len,K*100
        #log_var = clamp_preserve_gradients(log_var, -10.0, 1.0)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        kl = self.kl_divergence(z, mu, std)*mask
        # decoded
        ne_hat = self.ne_decoder1(self.relu2(self.ne_decoder(z)))
        decoder_mu = self.decoder_mu(ne_hat)
        decoder_std = self.decoder_std(ne_hat)
        ne_hat =decoder_mu
        # reconstruction loss
        #recon_loss = self.gaussian_likelihood(ne_hat, self.ne_log_scale, Y_temp)
        #recon_loss =self.gaussian_likelihood(decoder_mu, decoder_std, Y_temp)
        #self.ne_log_scale = clamp_preserve_gradients(self.ne_log_scale, -3.0, 3.0)
        #recon_loss = self.event_mse(ne_hat,Y_temp)
        recon_loss = self.mse_loss(ne_hat,Y_temp)
        recon_loss_tpp = recon_loss
        recon_loss = recon_loss.sum(-1)*mask
        elbo = kl_weight*kl + recon_loss  ### recon_loss 
        num_events = mask.sum()

        elbo = elbo.sum()/num_events
        log_dict = {
            'elbo': elbo.item(),
            'kl': (kl.sum()/num_events).item(),
            'reconstruction': (recon_loss.sum()/num_events).item(),
        }

        #Y = self.embedding_hidden(Y)
        X = torch.cat((X,ne_hat.view(-1,ne_hat.shape[2])),-1)

        #X = torch.cat((X,Y),-1)
        #print(X.shape)
        X = self.sigmactivation(X)
        X = self.hidden_to_hidden_time(X)
        X = self.sigmactivation(X)
        X = X.view(batch_size,seq_len,X.shape[-1])  
        self.inter_time_dist = self.lognormalmix.get_inter_time_dist(X) #### X is context 
        YDelta = YDelta.clamp(1e-10)
        inter_time_log_loss = self.inter_time_dist.log_prob(YDelta)   ### batch_size*seq_len
        
        return ne_hat,inter_time_log_loss,elbo,log_dict,Y_clusterid ##,log_dict,std,mu,z,recon_loss_tpp,decoder_mu,decoder_std



def get_event_prediction_rate(Y,Y_hat): 
    ### Assumes pad in Y is -1
    ### Y_hat is unnormalized weights
    mask = Y!=-1
    num_events = mask.sum()
    Y_hat = torch.argmax(Y_hat,dim=1)
    true_predicted = (Y_hat == Y)*mask
    true_predicted= true_predicted.sum()
    return true_predicted.item()*1.00/num_events.item()

def get_time_mse(T,T_hat,Y):
    mask = Y!=-1
    num_events = mask.sum()
    diff = (T-T_hat)*mask
    diff = diff*diff
    return diff.sum()/num_events
    
def get_topk_event_prediction_rate(Y, Y_hat, k=5, ignore_Y_value = -1): 
    ### Assumes pad in Y is -1
    ### Y_hat is unnormalized weights
    mask = Y!=ignore_Y_value
    num_events = mask.sum().item()
    #print(num_events)
    #Y_topk = torch.topk(Y_hat,k= k,dim=-1,largest=True)
    #Y_topk = Y_topk.indices.detach().cpu().numpy()
    #Y_cpu = Y.detach().cpu().numpy()
    true_predicted = sum([1 for i,item in enumerate(Y) if item != ignore_Y_value and item in Y_hat[i][:k]])
    return true_predicted*1.00/num_events

def get_X_Y_T_CID_from_sequences(sequences):  ### This also need to provide the cluster id of the 
    seq_X = []
    seq_Y = []
    seq_Xt = []
    seq_Yt = []
    seq_XDelta = []
    seq_YDelta = []
    seq_XCID = []
    seq_YCID = []
    for seq in sequences:
        seq_X.append([item[0] for item in seq[:-1]])  ## O contain node id
        seq_Y.append([item[0] for item in seq[1:]])
        seq_Xt.append([item[1] for item in seq[:-1]])   ## 1 contain timestamp
        seq_Yt.append([item[1] for item in seq[1:]])
        seq_XDelta.append([item[2] for item in seq[:-1]])   ## 2 contain delta from previous event
        seq_YDelta.append([item[2] for item in seq[1:]])
        seq_XCID.append([item[3] for item in seq[:-1]])   ## 3 contains the cluster id
        seq_YCID.append([item[3] for item in seq[1:]])
    X_lengths = [len(sentence) for sentence in seq_X]
    Y_lengths = [len(sentence) for sentence in seq_Y]
    max_len = max(X_lengths)
    return seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len,seq_XCID,seq_YCID
#seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,max_len,seq_XCID,seq_YCID = get_X_Y_T_CID_from_sequences(sequences)



def data_shuffle(seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,seq_XCID,seq_YCID):
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
    seq_XCID = [seq_XCID[i] for i in indices]
    seq_YCID = [seq_YCID[i] for i in indices]
    return seq_X,seq_Y,seq_Xt,seq_Yt,seq_XDelta,seq_YDelta,X_lengths,Y_lengths,seq_XCID,seq_YCID


class EdgeNodeLSTM(nn.Module):
    def __init__(self, vocab, node_pretrained_embedding, nb_layers, num_components, edge_emb_dim, nb_lstm_units=100, clust_dim=3, 
                 batch_size=3,device='cpu'):
        super(EdgeNodeLSTM, self).__init__()
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers  # number of LSTM layers
        self.nb_lstm_units = nb_lstm_units  # dimension of hidden layer h
        self.clust_dim = clust_dim  # dimension of the cluster embedding
        self.batch_size = batch_size
        self.edge_emb_dim = edge_emb_dim  # number of edge features
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1
        self.gnn_dim = node_pretrained_embedding.shape[1]
        self.edge_emb_dim = edge_emb_dim
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
            input_size=self.clust_dim + self.gnn_dim + self.edge_emb_dim,   ## cluster + GNN + edge embedding
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
        self.edge_decoder2 = nn.Linear(128,self.edge_emb_dim)  # layer 2 gnn_decoder
        self.edge_decoder3 = nn.Linear(self.edge_emb_dim, self.edge_emb_dim)  # layer 3 gnn_decoder
                
        self.mse_los_gnn = nn.MSELoss(reduction='none')
        self.mse_loss_edge = nn.MSELoss(reduction='none')
        self.celoss_cluster = nn.CrossEntropyLoss(ignore_index=0)

        self.relu_cluster = nn.LeakyReLU()  # activation forhidden to determin cluster id
        self.relu_edge = nn.LeakyReLU()  #activation for reconstruction edge attributes
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

    def forward(self, X, Y, Xedge, Yedge, X_lengths, mask, kl_weight, XCID, YCID):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        batch_size, seq_len = X.size()
        
        # ---------------------
        # 1. embed the input
        # --------------------
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)

        X = self.gnn_embedding(X)  # retrieve the gnn embedding from node vocab id
        X = self.embedding_hidden(X)  #TODO WHY?
        XCID_embedding = self.cluster_embeddings(XCID)  # retrieve embedding for cluster id
        X = torch.cat((Xedge, X, XCID_embedding), -1)
        
        
        # ---------------------
        # 2. Run through RNN
        # ---------------------
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True,enforce_sorted=False)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # ---------------------
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        X = X.contiguous()
        # X = X.view(-1, X.shape[2]) #reshape the data so it goes into the linear layer
        
        # Predict cluster id props
        Y_hat = self.hidden_to_ne_hidden(X)  # Z_cluster
        Y_hat = self.relu_cluster(Y_hat) ### Introducing non-linearity
        # Y_hat = Y_hat.view(batch_size, seq_len,Y_hat.shape[-1])
        Y_clusterid = self.clusterid_hidden(Y_hat)  # prop distrubution over clusters.
    
        
        #  get the mu and variance parameters voor retrieving GNN embedding
        #TODO need to understand why Y_hat is used as input and not X
        mu, log_var = self.cluster_mu(Y_hat), self.cluster_var(Y_hat)
        mu = mu.view(batch_size,seq_len,self.num_components,self.mu_hidden_dim)
        log_var = log_var.view((batch_size,seq_len,self.num_components,self.mu_hidden_dim))
        
        #YCID is expected to have size of batch_size*seq_len
        if self.training:  # select true cluster during training?
            Y_clusterid_sampled = YCID.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)
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
        ne_hat = self.gnn_decoder3(ne_hat)  #de layer decoder gnn embedding
        
        # Reconstruct gnn embedding
        edge_hat = self.edge_decoder2(self.relu_edge(self.edge_decoder1(z)))  # reconstruct  GNN embeding
        edge_hat = self.edge_decoder3(edge_hat)  #de layer decoder gnn embedding
        
        # prep y_true
        Y = self.gnn_embedding(Y)
        Y_temp = Y  #### Will be used to calculate the elbo loss
        Y = Y.view(-1,Y.shape[2])
        
        # reconstruction loss
        kl = self.kl_divergence(z, mu, std)*mask   # used for regularisation
        recon_loss_ne = self.mse_los_gnn(ne_hat,Y_temp)
        recon_loss_ne = recon_loss_ne.sum(-1)*mask
        recon_loss_edge = self.mse_loss_edge(edge_hat,Yedge)
        recon_loss_edge = recon_loss_edge.sum(-1)*mask
        elbo = kl_weight*kl + recon_loss_ne + recon_loss_edge  ### recon_loss 
        num_events = mask.sum()
        elbo = elbo.sum()/num_events
        
        #cluster loss
        Y_clusterid = Y_clusterid.view(-1, Y_clusterid.shape[-1])
        loss_cluster = self.celoss_cluster(Y_clusterid,YCID.view(-1))
        
        loss = elbo + loss_cluster
        
        log_dict = {
            'elbo': elbo.item(),
            'kl': (kl.sum()/num_events).item(),
            'reconstruction_ne': (recon_loss_ne.sum()/num_events).item(),
            'reconstruction_edge': (recon_loss_edge.sum()/num_events).item(),
            'cross_entropy_cluster': (loss_cluster.sum()/num_events).item(),
        }      
        
        
        return ne_hat, edge_hat, loss, log_dict, Y_clusterid
