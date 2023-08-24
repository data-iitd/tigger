import unittest
import os
import sys
import pandas as pd
import numpy as np
print(f"current idr: {os.getcwd()}")
sys.path.append(os.getcwd())

from tigger_package.inductive_controller import InductiveController

class InductiveControllerTest(unittest.TestCase):
    
    
    def setUp(self):
        output_path = 'data/test_graph/'
        node_feature_path = output_path + 'test_node_attr.parquet'
        edge_list_path = output_path + "test_edge_list.parquet"
        graphsage_embeddings_path = output_path + 'test_embedding.pickle'
        self.n_walks=10
        self.l_w=6
        self.inductiveController = InductiveController(
            node_feature_path=node_feature_path,
            edge_list_path=edge_list_path,
            graphsage_embeddings_path=graphsage_embeddings_path,
            n_walks=self.n_walks,
            batch_size = 6,
            num_clusters = 5,
            l_w = self.l_w,
            verbose = 1
        )
    
    
    def test_input_edge_dims(self):
        edge_cols = self.inductiveController.edge_attr_cols
        edge_df_dim = self.inductiveController.data.shape
        self.assertEqual(len(edge_cols), 3, msg="invalid number of edge attributes")
        self.assertEqual(edge_df_dim, (10,5), msg="invalid edge df dimension")
        
    def test_edge_objects(self):
        number_of_edges = len(self.inductiveController.edges)
        self.assertEqual(number_of_edges, 10, msg="invalid number of edge objects")
        
        last_edge = self.inductiveController.edges[9]
        self.assertEqual(last_edge.start, 8, msg="incorrect start node for edge #10")
        self.assertEqual(last_edge.end, 10, msg="incorrect end node for edge #10")
        
        for edge in self.inductiveController.edges:
            self.assertEqual(edge.attributes[0], edge.start, msg="edge attribute 0 doesn't match start node id")
            self.assertEqual(edge.attributes[1], edge.end, msg="edge attribute 1 doesn't match start node id")
            self.assertEqual(edge.attributes[2], edge.start, msg="edge attribute 2 doesn't match start node id")
            if edge.end==8:
                self.assertEqual(len(edge.outgoing_edges), 2, msg="edge 7to 8 doesn't has 2 successive edges")
                self.assertEqual(edge.out_nbr_sample_probs[0], 0.5, msg="sample prob is incorrect for edge 7 to 8")
            elif edge.end > 8:
                self.assertEqual(len(edge.outgoing_edges), 0, msg="no outgoing edge expected")
            else:
                self.assertEqual(len(edge.outgoing_edges), 1, msg="only one outgoing edge expected")
                self.assertEqual(edge.out_nbr_sample_probs[0], 1, msg="sample prob is incorrect")
            
            for out_edge in edge.outgoing_edges:
                self.assertEqual(out_edge.start, edge.end, msg="outgoing edge start id does not match with edge end id")
                                 
            
        
    def test_node_dict(self):
        number_of_nodes = len(self.inductiveController.node_id_to_object)
        self.assertEqual(number_of_nodes, 11, msg="invalid number of node objects")
        
        last_node = self.inductiveController.node_id_to_object[10]
        self.assertEqual(last_node.id, 10, msg="last node has incorrect id")
        self.assertEqual(len(last_node.as_start_node), 0, msg="last node has incorrect number of as_start_node")
        self.assertEqual(len(last_node.as_end_node), 1, msg="last node has incorrect number of as_end_node")
        
        for node in self.inductiveController.node_id_to_object.values():
            if node.id != 8:
                self.assertTrue(len(node.as_start_node)<2, msg="to many start node occurences")
            self.assertTrue(len(node.as_end_node)<2, msg="to many end node occurences")
            
            if len(node.as_start_node)>0:
                self.assertTrue(node.as_start_node[0].start==node.id, msg="mismatch in node id")
                
            if len(node.as_end_node)>0:
                self.assertTrue(node.as_end_node[0].end==node.id, msg="mismatch in node id")
                
    def test_vocab(self):
        vocab = self.inductiveController.vocab
        
        self.assertTrue(vocab['<PAD>']==0, msg="wrong pad id")
        self.assertTrue(vocab['end_node']==1, msg="wrong end note id")
        for node in self.inductiveController.node_id_to_object.values():
            id = node.id
            vocab_id = vocab[id]
            self.assertTrue(id+2==vocab_id, msg="mismatch between node and vocab id")
            
    def test_node_features(self):
        node_features = self.inductiveController.node_features
        vocab = self.inductiveController.vocab
        dims = node_features.shape
        # 11 nodes + 2 for padding + end node
        self.assertEqual(dims, (13,3), msg="incorrect feature dims")
        
        #check values for padding and end node
        self.assertEqual(node_features.iloc[vocab['<PAD>'],0], 0, msg="incorrect attr value for padding")
        self.assertEqual(node_features.iloc[vocab['end_node'],0], 0, msg="incorrect attr value for end node")
        
        for node_id in range(10):
            self.assertEqual(node_features.iloc[vocab[node_id],0], node_id, msg="incorrect attr value for nodes")
            
    def test_node_embedding(self):
        embed = self.inductiveController.node_embedding_matrix
        vocab = self.inductiveController.vocab
        dims = embed.shape
        
        self.assertEqual(dims, (13,16), msg="incorrect embedding dims")
        
        #check values for padding and end node
        self.assertEqual(sum(embed[vocab['<PAD>']]), 0, msg="incorrect embed value for padding")
        self.assertEqual(sum(embed[vocab['end_node']]), 16, msg="incorrect embed value for end node")
        
        for node_id in range(10):
            self.assertAlmostEqual(sum(embed[vocab[node_id]]), node_id * 2 / 11 * 16, msg="incorrect embed value for nodes")
            
    def test_cluster_labels(self):
        cl = self.inductiveController.cluster_labels
        vocab = self.inductiveController.vocab
        
        #check values for padding and end node
        self.assertEqual(cl[vocab['<PAD>']], 0, msg="incorrect cluster for padding")
        self.assertEqual(cl[vocab['end_node']], 1, msg="incorrect cluster for end node")
        
        number_of_cl = len(set(cl))
        self.assertEqual(number_of_cl, 7, msg="incorrect number of clusters")
        self.assertNotIn(0, cl[2:], msg="padding cluster used for node")
        self.assertNotIn(1, cl[2:], msg="end node cluster used for node")
        
    def test_random_walks(self):
        rws = self.inductiveController.sample_random_Walks()
        vocab = self.inductiveController.vocab
        
        self.assertTrue(len(rws) <= self.n_walks, msg="incorrect number of random walks")
        
        for rw in rws:
            self.assertTrue(len(rw) <= self.l_w + 1, msg="too many steps in random walk")
            self.assertTrue(len(rw) > 1, msg="too few steps in random walk")
            
            for step in rw:
                edge_attr = step[0]
                embed = step[1]
                cluster_id = step[2]
                node_attr = step[3]
                node_id = step[3][0]  # first value of node feature
                
                if cluster_id != vocab['end_node']:
                    self.assertTrue(edge_attr[1]== node_id, msg="middle edge feature is different from end node id")
                    self.assertTrue(len(embed) == 16, msg="vocab id doesn't match with node id")
                    self.assertTrue(cluster_id>1, msg="cluster id is less then 2")
                    
                else:
                    self.assertTrue(sum(edge_attr)==3, msg="attributes of end edge are not null")
                    self.assertTrue(step == rw[-1], msg="end node not at the end")
                    self.assertTrue(sum(embed)==16, msg="incorrect clustre id for end node")
                    self.assertTrue(sum(node_attr)==3, msg="attributes of end node are not null")
                    
    def test_x_y_sequences(self):
        rws = self.inductiveController.sample_random_Walks()
        seqs = self.inductiveController.get_X_Y_from_sequences(rws)
        
        for i, rw in enumerate(rws):
            for j, step in enumerate(rw):
                self.assertEquals(step[0], seqs['edge_attr'][i][j], msg="mismatch in edge attributes")
                self.assertEquals(step[1], seqs['node_embed'][i][j], msg="mismatch in node_embed")
                self.assertEqual(step[2], seqs['cluster_id'][i][j], msg="mismatch in cluster id")
                self.assertEquals(step[3], seqs['node_attr'][i][j], msg="mismatch in node attributes")
                
            steps = len(rw)
            self.assertEqual(steps, seqs['x_length'][i], msg="mitmatch in walk length")
            
    def test_get_batch(self):
        rws = self.inductiveController.sample_random_Walks()
        seqs = self.inductiveController.get_X_Y_from_sequences(rws)
        batch = self.inductiveController.get_batch(0, 6 , seqs)
        
        for i in range(6):  # single sequence
            rw = rws[i]
            for j, step in enumerate(rw):
                self.assertEquals(step[0], batch['edge_attr'][i][j].tolist(), msg="mismatch in edge attributes")
                self.assertEqual(step[1], batch['node_embed'][i][j].tolist(), msg="mismatch in node_embed")
                self.assertEqual(step[2], batch['cluster_id'][i][j].tolist(), msg="mismatch in cluster id")
                self.assertEquals(step[3], batch['node_attr'][i][j].tolist(), msg="mismatch in node attributes")
                
            steps = len(rw)
            self.assertEqual(steps-1, batch['x_length'][i], msg="mitmatch in walk length")
            
            # check batch values
            for j in range(steps, 6):
                self.assertEquals([0, 0, 0], batch['edge_attr'][i][j].tolist(), msg="mismatch in edge padding")
                self.assertEquals([0]*16, batch['node_embed'][i][j].tolist(), msg="mismatch in node_embed padding")
                self.assertEqual(0, batch['cluster_id'][i][j].tolist(), msg="mismatch in cluster padding")
                self.assertEquals([0, 0, 0], batch['node_attr'][i][j].tolist(), msg="mismatch in node padding")
            
            # check overall lengths  
            self.assertEqual(list(batch['edge_attr'][0].size()), [7, 3], msg="mismatch in edge length")
            self.assertEqual(list(batch['node_embed'][0].size()), [7, 16], msg="mismatch in node_embed length")
            self.assertEqual(list(batch['cluster_id'][0].size()), [7], msg="mismatch in cluster length")
            self.assertEqual(list(batch['node_attr'][0].size()), [7, 3], msg="mismatch in node length")  
            
    def test_synthetic_nodes_to_seqs(self):
        nodes = pd.read_parquet("data/test_graph/synth_nodes.parquet")
        seqs = self.inductiveController.synthetic_nodes_to_seqs(nodes.iloc[:10, :])
        
        
        self.assertAlmostEqual(sum(seqs['node_embed'][0][0]), 0.1*16, msg="mismatch in embedding")
        self.assertAlmostEqual(sum(seqs['node_attr'][0][0]), 0.2*3, msg="mismatch in node_attr")
        self.assertAlmostEqual(sum(seqs['edge_attr'][0][0]), 0.3*3, msg="mismatch in edge_attr")
        
        self.assertAlmostEqual(np.sum(np.array(seqs['node_embed'])), 0.1*16*10, msg="mismatch in embedding")
        self.assertAlmostEqual(np.sum(np.array(seqs['node_attr'])), 0.2*3*10, msg="mismatch in node_Attr")
        self.assertAlmostEqual(np.sum(np.array(seqs['edge_attr'])), 0.3*3*10, msg="mismatch in edge_attr")
        
        