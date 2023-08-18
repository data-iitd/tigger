import unittest
import os
print(f"current idr: {os.getcwd()}")
# from tigger_package.inductive_controller import InductiveController

import sys
sys.path.append('/Users/tonpoppe/workspace/tigger_adj_rep/tigger_adj')


import tigger_package as tp

class InductiveControllerTest(unittest.TestCase):
    
    
    def test_loading_edges(self):
        output_path = 'data/test_graph/'
        node_feature_path = output_path + 'test_node_attr.parquet'
        edge_list_path = output_path + "test_edge_list.parquet"
        graphsage_embeddings_path = output_path + 'test_embedding.pickle'
        n_walks=10
        inductiveController = InductiveController(
            node_feature_path=node_feature_path,
            edge_list_path=edge_list_path,
            graphsage_embeddings_path=graphsage_embeddings_path,
            n_walks=n_walks,
            batch_size = 6,
            num_clusters = 5,
            l_w = 7
        )
        
        edge_cols = inductiveController.edge_attr_cols
        self.assertEqual(len(edge_cols), 3)
        