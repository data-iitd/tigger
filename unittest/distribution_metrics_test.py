import unittest
import os
import sys
import pandas as pd
import numpy as np
print(f"current idr: {os.getcwd()}")
sys.path.append(os.getcwd())

from tigger_package.orchestrator import Orchestrator
from tigger_package.metrics.distribution_metrics import NodeDistributionMetrics, EdgeDistributionMetrics

class NodeDistributionMetricsTest(unittest.TestCase):
    
    def setUp(self):
        base_folder = 'unittest/test_data/test_graph/'
        self.orchestrator = Orchestrator(base_folder)
        nodes = self.orchestrator._load_nodes()
        synth_nodes = nodes.copy()
        synth_nodes.iloc[10,:]= [7, 8, 10]
        self.ndm = NodeDistributionMetrics(nodes, synth_nodes)
        
    def test_calculate_wasserstein_distance(self):
        ws_df = self.ndm.calculate_wasserstein_distance()
        
        self.assertAlmostEqual(ws_df.iloc[0,0] , 0.272727, places=3, msg="wasserstein distance for attr 0 is not correct")
        self.assertAlmostEqual(ws_df.iloc[1,0] , 0.181818, places=3, msg="wasserstein distance for attr 1 is not correct")
        self.assertAlmostEqual(ws_df.iloc[2,0] , 0, msg="wasserstein distance for attr 2 is not correct")
        
class EdgeDistributionMetricsTest(unittest.TestCase):
    def setUp(self):
        base_folder = 'unittest/test_data/test_graph/'
        self.orchestrator = Orchestrator(base_folder)
        edges = self.orchestrator._load_edges()
        synth_edges = edges.copy()
        synth_edges.iloc[9] = [7, 9, 9, 9, 8]
        
        synth_edges = synth_edges.rename({'start': 'src', 'end': 'dst'}, axis=1)
        self.edm = EdgeDistributionMetrics(edges, synth_edges)
        
    def test_calculate_wasserstein_distance(self):
        ws_df = self.edm.calculate_wasserstein_distance()
        
        self.assertAlmostEqual(ws_df.iloc[0,0] , 0.1, places=3, msg="wasserstein distance for attr 0 is not correct")
        self.assertAlmostEqual(ws_df.iloc[1,0] , 0.1, places=3, msg="wasserstein distance for attr 1 is not correct")
        self.assertAlmostEqual(ws_df.iloc[2,0] , 0, msg="wasserstein distance for attr 2 is not correct")
        
    def test_get_degrees_dist(self):
        self.edm.get_degrees_dist()
        
        edges_out = self.edm.edges_degree['out_degree']
        self.assertEqual(edges_out.iloc[0] , 8, msg="edge outdegree = 1 for edges is not correct")
        self.assertEqual(edges_out.iloc[1] , 1, msg="edge outdegree = 2 for edges is not correct")
        
        edges_in = self.edm.edges_degree['in_degree']
        self.assertEqual(edges_in.iloc[0] , 10, msg="edge indegree = 1 for edges is not correct")
        
        edges_out = self.edm.synth_edges_degree['out_degree']
        self.assertEqual(edges_out.iloc[0] , 8, msg="synth edge outdegree = 1 for edges is not correct")
        self.assertEqual(edges_out.iloc[1] , 1, msg="synth edge outdegree = 2 for edges is not correct")
        
        edges_in = self.edm.synth_edges_degree['in_degree']
        self.assertEqual(edges_in.iloc[0] , 8, msg="synth edge indegree = 1 for edges is not correct")
        self.assertEqual(edges_out.iloc[1] , 1, msg="synth edge indegree = 2 for edges is not correct")
        
    def test_get_degree_wasserstein_distance(self):
        ws_df = self.edm.get_degree_wasserstein_distance()   
        
        self.assertAlmostEqual(ws_df.iloc[0,0] , 1.5, places=3, msg="wasserstein distance for indegree is not correct")
        self.assertAlmostEqual(ws_df.iloc[1,0] , 0, places=3, msg="wasserstein distance outdegree is not correct")
        
    def test_widgets_distr(self):
        self.edm.gtrie_dir = '~/Downloads/gtrieScanner_src_01/'
        self.edm.temp_dir = 'temp/'
        df, mean = self.edm.widgets_distr()
        
        self.assertAlmostEqual(mean, 0.0170940170940171, places=3, msg="mean fraction of widget deviates")
        
    def test_clustering_coef_undirected(self):
        # create graphs and edm object
        edges = pd.DataFrame([[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]], columns=['start', 'end'])
        synth_edges = pd.DataFrame([[1,2], [1,3], [1,4], [2,3], [3,4]], columns=['src', 'dst'])
        edm = EdgeDistributionMetrics(edges, synth_edges)
        
        edm.gtrie_dir = '~/Downloads/gtrieScanner_src_01/'
        edm.temp_dir = 'temp/'
        # df, mean = edm.widgets_distr()
        cc = edm.clustering_coef_undirected()
        
        self.assertAlmostEqual(cc['edges'], 1, places=5, msg="cluster coef for edges deviates")
        self.assertAlmostEqual(cc['synth_edges'], 0.75, places=5, msg="cluster coef for edges deviates")
      