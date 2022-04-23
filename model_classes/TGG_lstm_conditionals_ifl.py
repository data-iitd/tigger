
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from Utils import alias_setup,alias_draw
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
import transformer.Constants as Constants
import Utils
from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
import csv
from torch.autograd import Variable
from torch.nn import functional as F


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
    def __init__(self, vocab,nb_layers, nb_lstm_units=100, embedding_dim=3, batch_size=3,time_emb_dim=10,device='cpu'):
        super(EventLSTM, self).__init__()
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.time_emb_dim = time_emb_dim
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1

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
        self.hidden_to_events = nn.Linear(self.nb_lstm_units, self.nb_events)
        #self.hiddent_
        self.hidden_to_hidden_time = nn.Linear(self.nb_lstm_units+self.embedding_dim,100)
        self.hidden_to_time = nn.Linear(100,1)
        self.sigmactivation = nn.Sigmoid()
        self.device = device
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units).to(self.device)
        hidden_b = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units).to(self.device)

        return (hidden_a, hidden_b)

    def forward(self, X,Y,Xt,XDelta, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        batch_size, seq_len = X.size()
        
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)

        X = self.word_embedding(X)
        Y = self.word_embedding(Y)
        #print(X.shape,Y.shape)
        #Xt = Xt.view(-1,1)
        #Xt = self.time_to_embedding(Xt)
        #Xt = Xt.view(batch_size,seq_len,Xt.shape[-1])
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
        Y = Y.view(-1,Y.shape[2])
        #print(X.shape,Y.shape)
        
        # run through actual Event linear layer
        Y_hat = self.hidden_to_events(X)
        Y_hat = Y_hat.view(batch_size, seq_len, self.nb_events)
        
        X = torch.cat((X,Y),-1)
        X = self.sigmactivation(X)
        X = self.hidden_to_hidden_time(X)
        X = self.sigmactivation(X)
        #print(X.shape)
        # Run through actual Time Linear layer
        T_hat = self.hidden_to_time(X)
        T_hat = T_hat.view(batch_size, seq_len, 1)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        #X = F.log_softmax(X, dim=1)
        #print(X.shape)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        return Y_hat,T_hat

  
    