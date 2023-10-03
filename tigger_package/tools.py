import pandas as pd
import networkx as nx


def nx_to_df(graph):
    edges = nx.to_pandas_edgelist(graph)
    nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    nodes["id"] = nodes.index
    return (nodes, edges)

def plot_adj_matrix(adj_df):
    attr = list(adj_df.columns)
    attr.remove('dst')
    attr.remove('src')
    G = nx.from_pandas_edgelist(adj_df, source='src', target='dst', edge_attr=['cnt_forward', 'size'], create_using=nx.DiGraph)
    nx.draw(G)