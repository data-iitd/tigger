# TIGGER: Scalable Generative Modelling for Temporal Interaction Graphs

This repository contains the code of paper [TIGGER: Scalable Generative Modelling for Temporal Interaction Graphs](https://www.cse.iitd.ac.in/~srikanta/publication/aaai-22-b/aaai-22-b.pdf) to appear in main track of AAAI-2022. The code has been run and tested on Python 3.7.6 and Ubuntu 16.04.5 LTS (GNU/Linux 4.15.0-45-generic x86_64).   


### Abstract
There has been a recent surge in *learning generative models* for graphs. While impressive progress has been made on static graphs, work on generative modeling of *temporal graphs* is at a nascent stage with significant scope for improvement. First, existing generative models do not scale with either the time horizon or the number of nodes. Second, existing techniques are *transductive* in nature and thus do not facilitate knowledge transfer. Finally, due to their reliance on one-to-one node mapping from source to the generated graph, existing models leak node identity information and do not allow *up-scaling/down-scaling* the source graph size. In this paper, we bridge these gaps with a novel generative model called TIGGER. TIGGER derives its power through a combination of *temporal point processes* with *auto-regressive* modeling enabling both transductive and inductive variants. Through extensive experiments on real datasets, we establish TIGGER generates graphs of superior fidelity, while also being up to 3 orders of magnitude faster than the state-of-the-art.


## Citation

```
@inproceedings{gupta2022tigger,
 acceptance = {0.15},
 author = {Shubham Gupta and Sahil Manchanda and Sayan Ranu and Srikanta Bedathur},
 booktitle = {Proc. of the 36th AAAI Conference on Artificial Intelligence (AAAI)},
 title = {TIGGER: Scalable Generative Modelling for Temporal Interaction Graphs (to appear)},
 year = {2022}
}

```

*** 
### Code Description


> **train_transductive.py** is the training file for transductive model.

> **test_inductive.py** is code for sampling inductive random walks.

> **train_inductive.py** is training file for inductive model.  

> **test_transductive.py** is code for sampling inductive random walks.

> **graph_generation_from_sampled_random_walks.py** is used for assembling synthetic temporal graph from sampled random walks.  

> **graph_metrics.py** is used to compute the metrics for sampled graphs wrt input temporal graph.

### Important parameters wrt to code execution
We encourage users to read the paper thoroughly to get better understanding of some important hyper-parameters.

- data_path = This is the location of data file which should be in csv format having three columns \<start\>, \<end\> and \<days\>. \<start\> and <end\> are the nodes of each interaction and \<days\> denotes the time of corresponding interaction.   

- config_path = This is the name of the configuration folder which needs to be specified during training. Every model information and sampled temporal graphs will automatically be saved here. If folder is not present, it will created during execution. This folder is used consequently in next steps.

- num_epochs = No. of training epochs to train the generative model.

- filter_walk = This is the minimum length of random walks to be kept for training. 

- nb_layers = No. of layers in (EventLSTM defined in model_classes/transductive_model, EventClusterLSTM defined in model_classes/inductive_model).

- nb_lstm_units = No. of units in a LSTM Cell.

- time_emb_dim(d_T) = Required embedding size applying after applying Time2Vec transformation.

- num_mix_components(C) = No. of components in log normal mixture model which is used to predicting next time.

- window_interactions = This is the **W** parameter as describe in appendix. We usually keep it between 3 for small datasets to 6 for large datasets. 

- random_walk_sampling_rate = No. of random walks to be sampled. Heuristically, we keep it as (10/20)*M where M is total no. of interactions in input graph.

- l_w = length of expected random walk. During training we keep it around (10-20). During testing, we keep it ~2 for small datasets and higher ~ 6-15 for large datasets. 

- num_of_sampled_graphs= No. of graphs to be sampled.

- batch_size (train_\*.py) = Batch size during 1 forward call of a EventLSTM/EventClusterLSTM Cell.
- batch_size(in test_\*.py) = Batch size during 1 forward pass. We keep it 1024 for smaller datasets and 5K-50K for large size datasets. We note that given the capacity of GPU, keeping it higher results in faster sampling.

- Specific to inductive settings
    - num_clusters(K) : No. of clusters required during pre-processing each node embedding.
    - graph_sage_embedding_path : Path of trained graph-sage embeddings. Please note that code expects a dictionary where each key is a node id and corresponding value is node embedding.
    - gan_embedding_path: Path of embeddings sampled using a trained GAN. 

### Commands for TIGGER (Transductive)
```
cd <to_code_directory>  

python train_transductive.py --data_path=<data_path> --gpu_num=2 --config_path=<model_directory>  

python test_transductive.py --data_path=<data_path> --gpu_num=2 --config_path=<model_directory> --model_name=200 --random_walk_sampling_rate=20 --num_of_sampled_graphs=1  

python graph_generation_from_sampled_random_walks.py --data_path=<data_path> --config_path=<model_path> --num_of_sampled_graphs=1 --time_window=1 --topk_edge_sampling=1  

python graph_metrics.py --op=<original_graph_path> --sp=<sampled_graph_path>
```

#### Example syntax
```
python train_transductive.py --data_path=./data/CAW_data/wiki_744_50.csv --gpu_num=4 --config_path=models/test/

time python test_transductive.py --data_path=./data/CAW_data/wiki_744_50.csv --gpu_num=4 --config_path=models/test/ --model_name=200 --random_walk_sampling_rate=10 --num_of_sampled_graphs=1 --batch_size=1000   

time python graph_generation_from_sampled_random_walks.py --data_path=./data/CAW_data/wiki_744_50.csv --config_path=models/test/ --num_of_sampled_graphs=1 --time_window=1 --topk_edge_sampling=1 --l_w=2

python graph_metrics.py --op=models/test/results/original_graphs.pkl --sp=models/test/results/sampled_graph_0.pkl
```
### Commands for running TIGGER-I (Inductive)

We have adopted the GraphSage and WGAN training code from [git-hub](https://github.com/yizhidamiaomiao/DoppelgangerGraph) of [Generating a Doppelganger Graph: Resembling but Distinct](https://arxiv.org/pdf/2101.09593.pdf). Using graph-sage, we learn the node embeddings in data and using WGAN we learn a generative model which can be used to sample pre-specified batch of new node embeddings.

We are including the code here as well for completeness. Please check the graphsage/train_messaging.py and graphsage/sample_GAN.ipynb for the same. To train a graphsage model, we use the adjacency of each node as a feature vector. Additionally, for a bipartitie graphs like (wiki-edit), we add a additional 1 corresponding to self location of each node in the corresponding features( i.e. one-hot vector+adjacency vector). Please monitor the loss and pick the graph-sage model corresponding to best validation metric. Similarly, during training a GAN, monitor the displayed curves and pick the GAN model corresponding to curve having similar input and learnt distributions. 


```
cd <to_code_directory>
python train_inductive.py --data_path=<data_path> --gpu_num=2 --config_path=<model_directory> --graph_sage_embedding_path=<graphsage_embedding_path>
python test_inductive.py --data_path=<data_path> --gpu_num=2 --config_path=<model_directory> --model_name=90 --random_walk_sampling_rate=20 --num_of_sampled_graphs=1 --graph_sage_embedding_path=<graphsage_embedding_path> --gan_embedding_path=<path_of_embeddings_generated_by_gan> 
python graph_generation_from_sampled_random_walks.py --data_path=./data/CAW_data/wiki_744_50.csv --config_path=models/test/ --num_of_sampled_graphs=1 --time_window=1 --topk_edge_sampling=0 --l_w=<length_of_sampled_random_walk>
python graph_metrics.py --op=<original_graph_path> --sp=<sampled_graph_path>
```

#### Example syntax


```


python train_inductive.py --data_path=./data/CAW_data/wiki_744_50.csv --graph_sage_embedding_path=./graphsage_embeddings/wiki_50_744/embeddings.pkl --gpu_num=2 --config_path=models/test/ --num_epochs=1000 --window_interactions=3 --filter_walk=2 --l_w=10

time python test_inductive.py --data_path=./data/CAW_data/wiki_744_50.csv --gpu_num=2 --config_path=models/test/ --model_name=800 --random_walk_sampling_rate=10 --num_of_sampled_graphs=1 --graph_sage_embedding_path=./graphsage_embeddings/wiki_50_744/embeddings.pkl --gan_embedding_path=./graphsage_embeddings/wiki_50_744/gan_embeddings_N.npy --l_w=3

python graph_generation_from_sampled_random_walks.py --data_path=./data/CAW_data/wiki_744_50.csv  --config_path=models/test/ --num_of_sampled_graphs=1 --time_window=1 --topk_edge_sampling=0 --l_w=2

python graph_metrics.py --op=models/test/results/original_graphs.pkl --sp=models/test/results/sampled_graph_0.pkl
```
**We observe that varying/increasing the parameters of the tigger like num of levels of LSTM, size of an lstm unit, size of d_T, no. of components in log-normal mixture model,l_w, batch_size, random_walk_sampling_rate, gan-embeddings further improves the performance.** 


