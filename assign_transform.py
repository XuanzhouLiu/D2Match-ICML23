from enum import unique
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
from dataset import PairData

from torch_geometric.transforms import BaseTransform, OneHotDegree
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

from utils import to_graph_pair

class DegreeCheckAssign(BaseTransform):
    def __init__(self, neighbor_degree = True):
        self.neighbor_degree = neighbor_degree

    def __call__(self, data_pair):
        #print("on_call")
        graph_t, graph_s = to_graph_pair(data_pair)
        if hasattr(data_pair, "node_type_t"):
            node_type_t = data_pair.node_type_t.numpy()
            node_type_s = data_pair.node_type_s.numpy()
        else:
            node_type_t = np.ones(graph_t.number_of_nodes())
            node_type_s = np.ones(graph_s.number_of_nodes())
        
        assign = self.degree_mat(graph_t, graph_s, node_type_t, node_type_s)
        data_pair.degree_assign = assign
        return data_pair

    def degree_mat(self, graph_t, graph_s, node_type_t = None, node_type_s = None):
        num_t, num_s = graph_t.number_of_nodes(), graph_s.number_of_nodes()
        

        neigh_t, neigh_s = [], []
        for i in range(num_t):
            neigh_t.append(np.array(list(graph_t.neighbors(i))))
        for j in range(num_s):
            neigh_s.append(np.array(list(graph_s.neighbors(j))))

        A_t = nx.to_numpy_array(graph_t)
        A_s = nx.to_numpy_array(graph_s)

        if not node_type_s is None:
            num_t = np.sum(node_type_t == 1)
            num_s = np.sum(node_type_s == 1)
            assign = (node_type_t.reshape([-1,1]) == node_type_s.reshape([1,-1])).astype(float)
        else:
            assign = np.ones([num_t, num_s])

        for i in range(num_t):
            for j in range(num_s):
                if len(neigh_t[i]) == 0 and len(neigh_s[j]) > 0:
                    assign[i,j] = 0
                elif len(neigh_s[j]) == 0:
                    pass
                else:
                    if node_type_s is None:
                        deg_t = A_t[neigh_t[i], :][:, neigh_t[i]].sum(axis=1)
                        deg_s = A_s[neigh_s[j], :][:, neigh_s[j]].sum(axis=1)
                        mat = assign[neigh_t[i], :][:, neigh_s[j]] * (deg_t.reshape([-1,1]) >= deg_s.reshape([1,-1])).astype(float)
                        assign[i,j] *= (min(np.sum(mat, axis=0))>0)
                    else:
                        true_neigh_t = neigh_t[i][node_type_t[neigh_t[i]] == 1]
                        true_neigh_s = neigh_s[j][node_type_s[neigh_s[j]] == 1]
                        for type in np.unique(node_type_s, axis=0):
                            type_neigh_t = neigh_t[i][node_type_t[neigh_t[i]] == type]
                            type_neigh_s = neigh_s[j][node_type_s[neigh_s[j]] == type]
                            deg_t = A_t[true_neigh_t, :][:, type_neigh_t].sum(axis=1)
                            deg_s = A_s[true_neigh_s, :][:, type_neigh_s].sum(axis=1)
                            mat = (deg_t.reshape([-1,1]) >= deg_s.reshape([1,-1])).astype(float)
                            assign[i,j] *= (min(np.sum(mat, axis=0))>0)
        

        return assign

from chordless import ChordlessCycles
class ChordlessCycleTransfrom(BaseTransform):
    def __init__(self, length_list = [3,4]):
        self.length_list = length_list

    def __call__(self, data_pair):
        #print("on call")
        graph_t, graph_s = to_graph_pair(data_pair)
        y=data_pair.y
        t_nodes = graph_t.nodes()
        s_nodes = graph_s.nodes()

        nx.set_node_attributes(graph_t, { i:1 for i in t_nodes}, "node_type")
        nx.set_node_attributes(graph_s, { i:1 for i in s_nodes}, "node_type")

 
        graph_t = self.add_cycle_nodes(graph_t, t_nodes, self.length_list)
        graph_s = self.add_cycle_nodes(graph_s, s_nodes, self.length_list)

        data_t = pyg_utils.from_networkx(graph_t)
        data_s = pyg_utils.from_networkx(graph_s)

        '''
        x_t = torch.tensor(list(map(lambda x: x[1],list(graph_t.nodes(data="x")))), device=data_pair.x_t.device)
        x_s = torch.tensor(list(map(lambda x: x[1],list(graph_s.nodes(data="x")))), device=data_pair.x_s.device)

        type_t = F.one_hot(torch.tensor(list(map(lambda x: x[1],list(graph_t.nodes(data="node_type")))), device=x_t.device))
        type_s = F.one_hot(torch.tensor(list(map(lambda x: x[1],list(graph_s.nodes(data="node_type")))), device=x_s.device))
        '''

        data_pair = PairData(data_s, data_t)
        data_pair.node_type_t = data_t.node_type
        data_pair.node_type_s = data_s.node_type
        data_pair.y=y
        return data_pair

    def add_cycle_nodes(self, graph, node_list, length, default_x = None):
        if not isinstance(length, list):
            length = [length]
        subgraph = graph.subgraph(node_list)
        chordless_cycles = ChordlessCycles(subgraph, max(length))
        cycles = []
        for c in chordless_cycles:
            if len(c) in length:
                cycles.append(c)
        
        if default_x is None:
            default_x = np.ones(len(graph.nodes[0]["x"]))

        node_id = max(graph.nodes)+1
        for c in cycles:
            graph.add_node(node_id, x = default_x, node_type = len(c))
            for n in c:
                graph.add_edge(node_id, n)
            node_id+=1
        return graph

class NoFeatureTransfrom(BaseTransform):
    def __init__(self):
        super().__init__()
    
    def __call__(self, data_pair):
        data_pair.x_t = torch.ones(data_pair.x_t.size()).float()
        data_pair.x_s = torch.ones(data_pair.x_s.size()).float()
        return data_pair
