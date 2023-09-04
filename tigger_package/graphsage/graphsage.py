import torch
import math
import os
import random
import time
import statistics
import functools
from tigger_package.graphsage.encoders import Encoder
from tigger_package.graphsage.aggregators import MeanAggregator
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class GraphSAGE:
    
    def __init__(self, _N, adj_dic, feat_matrix, embedding_dim, level=2, verbose_level=4,
                 validation_fraction=0.1, batch_size=1024, dropout=0):
        self._N =_N
        self._M = 0  # this is set during training
        self.adj_dic=adj_dic  # node dict of neighbor list
        self.verbose = verbose_level  # 1) no output, 2) minimum output, 3)performance, 4) debug
        self.num_feat=feat_matrix.shape[1]  ### node type 0/ 1
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.dropout = dropout
        try: 
            if torch.backends.mps.is_available():
                device = torch.device("mps")
        except:
            if torch.cuda.is_available():
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
        self.enc1 = Encoder(self.features, self.num_feat, self.embedding_dim, self.adj_dic, self.agg1, gcn=False, device=self.device, dropout=self.dropout)
        self.agg2 = MeanAggregator(lambda nodes : self.enc1(nodes))
        self.enc2 = Encoder(lambda nodes : self.enc1(nodes), self.enc1.embed_dim, self.embedding_dim, self.adj_dic, self.agg2,base_model=self.enc1, 
                            gcn=False, device=self.device, dropout=self.dropout)
        if level == 1:
            self.graphsage = SupervisedGraphSage(self.enc1,self._N, device=self.device)
        else:
            self.graphsage = SupervisedGraphSage(self.enc2,self._N, device=self.device)
        
    
    def save_model(self,path='graph_graphsage.pth'):
        print("Saving to , ", path)
        torch.save(self.graphsage.state_dict(), path)
        
    def save_embedding(self, embedding_path='embeddings'):
        print(f"calculating and saving embeddings to {embedding_path}")
        np.save(embedding_path,self.get_embeddings())
        
    def load_model(self,path='graph_graphsage.pth'):
        self.graphsage.load_state_dict(torch.load(path), strict=False)
        # self.embedding_matrix_numpy = np.load(embedding_path).reshape((self._N,self.embedding_dim))
        
        
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
        embedding_matrix_numpy = None
        for i in range(math.ceil(self._N / 1024)):
            # get embedding for range
            ubound = min((i+1)*1024, self._N)
            embedding_matrix_torch = self.graphsage.enc(range(i*1024, ubound))
            if embedding_matrix_numpy is None:
                embedding_matrix_numpy = embedding_matrix_torch.detach().to('cpu').numpy()
            else:
                embedding_matrix_numpy = np.vstack(
                    (embedding_matrix_numpy, embedding_matrix_torch.detach().to('cpu').numpy())
                )
        return embedding_matrix_numpy

    def graphsage_train(self,boost_times=20,add_edges=1000,training_epoch=10000,
                        boost_epoch=5000,learning_rate=0.001,save_number=0,dirs='graphsage_model/'):
        
        train_stats = []  # list to keeptrack of the training results
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        
        # create train, validation and test split together with negative examples.
        datasets = self.get_train_validation_test_set(self.validation_fraction, 0.0)
        datasets_false, exempt_set = self.get_false_edges_datasets(1, 1, 10, datasets)
          
        train_dataset = np.concatenate([datasets['train_set'], datasets_false['train_set']])
        labels_dataset = np.concatenate([np.ones(len(datasets['train_set'])), np.zeros(len(datasets_false['train_set']))])
        val_dataset = np.concatenate([datasets['validation_set'], datasets_false['validation_set']])
        val_label = np.concatenate([np.ones(len(datasets['validation_set'])), np.zeros(len(datasets_false['validation_set']))])
        
        # train graphsage model
        # optimizer = torch.optim.Adam(self.graphsage.parameters(), lr=learning_rate, weight_decay=0)
        optimizer = torch.optim.AdamW(self.graphsage.parameters(), lr=learning_rate)
        train_metrics = self.graphsage.train(train_dataset,labels_dataset,training_epoch,optimizer, x_val=val_dataset, y_val=val_label, batch_size=self.batch_size)
        train_metrics['label'] = 'train'
        train_stats.append(train_metrics)
        self.save_model(path=dirs+'/graphsage'+str(save_number)+'.pth')
        # self.save_embedding(embedding_path=dirs+'/embedding_matrix'+str(save_number)+'.pth'))
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
                if len(add_train)==0:
                    continue
                    
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
            
            boost_metrics = self.graphsage.train(train_dataset, labels_dataset, boost_epoch, optimizer, x_val=val_dataset, y_val=val_label, batch_size=self.batch_size)
            boost_metrics['label'] = 'boost_'+str(boost_iter)
            train_stats.append(boost_metrics)
            self.save_model(path=dirs+'/graphsage'+str(save_number)+'_'+str(boost_iter)+'.pth')
            # self.save_embedding(embedding_path=dirs+'/embedding_matrix'+str(save_number)+'_'+str(boost_iter)+'.pth')
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
        print("create edge list")
        for src, neighbors in tqdm(self.adj_dic.items()):
            for dst in neighbors:
                if dst > src: # avoid duplicates and self loop because edges are in the dict in both directions
                    edges.append((src, dst))
                    
        self._M = len(edges)
        num_test = int(np.floor(self._M * test_frac)) # size of test set
        num_val = int(np.floor(self._M * validation_frac)) # size of train set
        assert (num_test + num_val) < (self._N * 0.8), "validation + test is greater 80% of total dataset"
        
              
        # create train, validation and test set
        print("shuffling")
        random.shuffle(edges)
        test_set = edges[:num_test]
        validation_set = edges[num_test: num_test+num_val]
        train_set = edges[num_test+num_val:]
        
        assert (len(test_set)+len(validation_set)+len(train_set) == self._M),  "length of datasets is unequal to the number of edges"  
        return {"train_set": train_set, "validation_set": validation_set, "test_set": test_set}
    
    def create_false_edges(self, size, exempt_set):
        """creates edges that are not in the edge list and neither in hte exempt set"""
        false_edges = []
        cnt = 0 
        while len(false_edges) < size and cnt < 2*size:
            cnt += 1
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




class SupervisedGraphSage(nn.Module):

    def __init__(self,enc,_N, device):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        embed_dim=self.enc.embed_dim
        self.device = device
        self.xent= nn.BCELoss().to(self.device)
        
        self.fc1 = nn.Linear(embed_dim,embed_dim).to(self.device)
        self.fc2 = nn.Linear(embed_dim,1).to(self.device)
        self._N=_N
    
    def forward(self,edges_list):
        node1=edges_list[:,0]
        node2=edges_list[:,1]
        x=self.enc(node1).to(self.device)
        y=self.enc(node2).to(self.device)
        
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
    
    def validation_step(self, x_val, y_val, batch_size):
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
        step = 0
        max_step = math.floor(y_val.shape[0] / batch_size)
        for epoch in range(epochs):
            # batch=random.sample(range(len(train)),batch_size)  #n time complexitiy!
            start_time = time.time()
            batch_edges = train[step*batch_size: (step+1)*batch_size]
            batch_labels = labels[step*batch_size: (step+1)*batch_size]
            step = step + 1 if step < (max_step-1) else 0  
            optimizer.zero_grad()
            loss = self.loss(batch_edges,Variable(torch.FloatTensor(batch_labels)).to(self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.005)
            optimizer.step()
            end_time = time.time()
            if epoch % 20==0:
                # print('\rEpoch:%d,Loss:%f,estimated time:%.2f'%(epoch, loss.item(),(end_time-start_time)*(epochs-epoch)),end="")
                print(f'\rEpoch: {epoch}, loss: {loss.item()}, elapsed time {end_time-start_time:.2f}, remaining time: {(end_time-start_time)*(epochs-epoch):.1f}', end="")
                train_loss.append(loss.item())
                epoch_id.append(epoch)
            if epoch%100==0 and x_val is not None and y_val is not None:
                val_loss.append(self.validation_step(x_val, y_val, batch_size=batch_size))
                val_epoch.append(epoch)
                # print('\n acc:'+str(self.train_acc(train,labels)))
        return {'train_loss': train_loss, 'epoch': epoch_id, 'val_loss': val_loss, 'val_epoch': val_epoch}
