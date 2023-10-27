#%%
# --------------------------------
# Create synthetic graph base on erdos renyi algo
# --------------------------------
'''
This notebook create a synthetic directed graph base on the erdos renyi algorithm.
The node attributes are normally distributed condition on the incoming or outgoing edge degree
Edge attribute are conditioned on the average incoming or outgoing degree.
'''
import random
import networkx as nx
node_cnt = 1000
#%%

def create_graph(node_cnt, node_att=8, edge_att=8):
    G = nx.fast_gnp_random_graph(node_cnt, p=0.1, seed=1, directed=True)
    random.seed(1)
    in_degree = {n: min(d, 10) for n,d in G.in_degree}
    out_degree = {n: min(d, 10) for n,d in G.out_degree}
    for i, att in enumerate(range(node_att)):
        if i%2==0:
            values = {n:random.normalvariate(d, d) for n, d in in_degree.items()}
        else:
            values = {n:random.normalvariate(d, d) for n, d in out_degree.items()}
            
        # normalize values with min /max
        max_val = max(values.values())
        min_val = min(values.values())
        values = {n: (v - min_val) / (max_val - min_val) for n,v in values.items()}
        nx.set_node_attributes(G,values, "attr"+str(att))
      
        
    # nx.set_edge_attributes(G, {i: random.random() for i in G.edges()}, "weight")
    for i, att in enumerate(range(edge_att-1)):
        if i%2==0:
            values =  {i: random.normalvariate(in_degree[i[0]]+out_degree[i[1]], in_degree[i[0]]+out_degree[i[1]]) for i in G.edges()}
        else:
            values =  {i: random.normalvariate(in_degree[i[1]]+out_degree[i[0]], in_degree[i[1]]+out_degree[i[0]]) for i in G.edges()}
     
        # normalize values with min /max
        max_val = max(values.values())
        min_val = min(values.values())
        values = {n: (v - min_val) / (max_val - min_val) for n,v in values.items()}
        nx.set_edge_attributes(G, values, "edge_att"+str(att))
    return G

G= create_graph(1000)

# %% Check values 

node_attr = pd.DataFrame([v for n,v in G.nodes(data=True)])
#%%

import matplotlib.pyplot as plt
import math
fig = plt.figure(figsize=(20,20))
dim = node_attr.shape[1]
for i, col in enumerate(node_attr.columns):
    ax = fig.add_subplot(math.ceil(dim/2),2, i+1)
    ax.hist(node_attr[col], bins=20, alpha=0.5, color='g', label='node'+str(i))

plt.legend()
plt.show()
# %%
