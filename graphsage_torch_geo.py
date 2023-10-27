#%%
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

x = torch.randn(8, 32)  # Node features of shape [num_nodes, num_features]
y = torch.randint(0, 4, (8, ))  # Node labels of shape [num_nodes]
edge_index = torch.tensor([
    [2, 3, 3, 6, 5, 6, 7],
    [0, 0, 1, 1, 2, 3, 4]],
)

#   0  1
#  / \/ \
# 2  3  |   4
# |   \ |   |
# 5     6   7

data = Data(x=x, y=y, edge_index=edge_index)


#%%
import torch_geometric as tg
loader = tg.loader.LinkNeighborLoader(
    data,
    edge_label_index = data.edge_index[:,:2],
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[1, 1],
    # Use a batch size of 128 for sampling training nodes
    batch_size=1,
    neg_sampling_ratio=1,
    shuffle=True
)

batch = next(iter(loader))
print(batch)


# %%
print(f"sampled edges:\n {data.edge_index[:,batch.e_id]}")
print(f"model input shape {batch.x.shape}")
print(f"{batch.edge_index}")
print(f"{batch.edge_label}")

for sampled_data in loader: 
    print(data.edge_index[:,sampled_data.e_id])
    print(f"{batch.edge_label}")
# %%


model = tg.nn.GraphSAGE(
    data.num_node_features,
    hidden_channels=4,
    num_layers=2,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# %%
import torch.nn.functional as nnf

def train():
    model.train()

    total_loss = 0
    for batch in loader:
        batch = batch
        print("begin batch")
        print(f"e_id: {batch.e_id}")
        print(data.edge_index[:,batch.e_id])
        print(f"relabelled edge_index:\n {batch.edge_index}")  
        print(f"edge_label_index:\n {batch.edge_label_index}")
        print(f"edge_label: \n {batch.edge_label}")
        
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        print(f"h shpe : {h.shape}")
        
        
        h_src = h[batch.edge_label_index[0]]
        print(f"h_src:\n {h_src}")
        h_dst = h[batch.edge_label_index[1]]
        print(f"h_dst:\n {h_dst}")
        pred = (h_src * h_dst).sum(dim=-1)
        print(f"pred:\n {pred}")
        loss = nnf.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        # optimizer.step()
        total_loss += float(loss) * pred.size(0)
        # print(f"edge_label_index:\n {batch.edge_label_index}")
        print(f"n_id: {batch.n_id}")
        print(f"x: {batch.x.shape}")

    return h
# %%
import time
times = []
for epoch in range(1, 2):
    start = time.time()
    print(f"H:\n {train()}")
    # val_acc, test_acc = test()
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
        #   f'Val: {val_acc:.4f}, Test: {test_acc:.4f}'
        #   )
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
# %%
transform = tg.transforms.RandomLinkSplit(
    num_val=0, num_test=0.4, is_undirected=False, 
    add_negative_train_samples=False, neg_sampling_ratio=0)
train_data, val_data, test_data = transform(data)

loader = tg.loader.LinkNeighborLoader(
    train_data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[1, 1],
    # Use a batch size of 128 for sampling training nodes
    batch_size=2,
    neg_sampling_ratio=1,
    shuffle=True
)
batch = next(iter(loader))

#%%

from torch_geometric.data import Data
from tigger_package.orchestrator import Orchestrator
import os
# os.chdir('..')
print(os.getcwd())

enron_folder = "data/enron/"
orchestrator = Orchestrator(enron_folder)

node_feature = orchestrator._load_nodes()
edges = orchestrator._load_edges()


#%%
edge_index = torch.tensor(edges[['start', 'end']].values)

data = Data(x = torch.tensor(node_feature.values), 
            edge_index=torch.transpose(edge_index, 0, 1))
#%%

# Add additional arguments to `data`:
# data.train_idx = torch.tensor([...], dtype=torch.long)
# data.test_mask = torch.tensor([...], dtype=torch.bool)

# Analyzing the graph structure:
data.num_nodes
data.is_directed()

# %%
import pandas as pd
path1 = "data/enron/embed/embedding.parquet" 
path2 = "data/enron/embed/embedding_final.parquet" 
df1 = pd.read_parquet(path1)
df1 =df1.reset_index(names='id')
df2 = pd.read_parquet(path2)
# %%
