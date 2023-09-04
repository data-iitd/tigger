import pandas as pd
import networkx as nx


def nx_to_df(graph):
    edges = nx.to_pandas_edgelist(graph)
    nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    nodes["id"] = nodes.index
    return (nodes, edges)