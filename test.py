#%% # create dummy feature matrixs
import numpy as np
import pandas as pd
import pickle

graphsage_embeddings_path = "graphsage_embeddings/bitcoin/embeddings.pkl"
node_embeddings_feature = pickle.load(open(graphsage_embeddings_path,"rb"))
print(f"dict size {len(node_embeddings_feature)}")
print(f"min key {min(node_embeddings_feature.keys())}")
print(f"max key {max(node_embeddings_feature.keys())}")
      

#%%
node_df = np.random.random(size=(3783, 11))
pdf = pd.DataFrame(node_df, columns = ["attr"+str(i) for i in range(11)])
pdf['id']=range(3783)
pdf.to_parquet("data/bitcoin/feature_attributes.parquet")



# %%
