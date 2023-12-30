# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import logging
import random
from networkx.algorithms import isomorphism
from utils import limit_time
import copy

MAX_MATCHING_TIME = 600

class Generator(object):
    def __init__(self, sizes, size_prob=None, edge_sizes=None, edge_size_prob=None):
        self.set_sizes(sizes, size_prob)
        self.set_edge_sizes(edge_sizes, edge_size_prob)

    def set_sizes(self, sizes, size_prob=None):
        self.sizes = sizes
        if sizes is not None:
            if size_prob is None:
                self.size_prob = np.ones(len(sizes)) / len(sizes)
            else:
                self.size_prob = size_prob

    def _get_size(self, size=None):
        if size is None:
            return np.random.choice(
                self.sizes, size=1, replace=True, p=self.size_prob
            )[0]
        else:
            return size

    def set_edge_sizes(self, edge_sizes=None, edge_size_prob=None):
        self.edge_sizes = edge_sizes
        if edge_sizes is not None:
            if edge_size_prob is None:
                self.edge_size_prob = np.ones(len(edge_sizes)) / len(edge_sizes)
            else:
                self.edge_size_prob = edge_size_prob

    def _get_edge_size(self, edge_size=None):
        if edge_size is None:
            if self.edge_sizes is None:
                return None
            return np.random.choice(
                self.edge_sizes, size=1, replace=True, p=self.edge_size_prob
            )[0]
        else:
            return edge_size
    
    def generate(self, size=None, edge_size=None):
        raise NotImplementedError
    

class GraphGenerator(Generator):
    def __init__(self,sizes, size_prob=None, feat_type=None, feat_dim = 0, edge_sizes=None, edge_size_prob=None, **kwargs):
        super(GraphGenerator, self).__init__(sizes, size_prob, edge_sizes, edge_size_prob)
        self.set_features(feat_type, feat_dim)
        
    def set_features(self, feat_type, feat_dim):
        self.feat_type = feat_type
        self.feat_dim = feat_dim
        
    def _generate_graph(self, size):
        return nx.Graph()
        
    def _generate_feat(self, size):
        if self.feat_type == None:
            return np.zeros([size,1]).astype(np.float32)
        elif self.feat_type == "random":
            return np.random.randn(size,self.feat_dim).astype(np.float32)
        elif self.feat_type == "ones":
            return np.ones([size,self.feat_dim]).astype(np.float32)
        return np.zeros(size,1).astype(np.float32)
    
    def modify_edges(self, graph, edge_size):      
        if edge_size-graph.number_of_edges()>0:
            non_edges = list(nx.non_edges(graph))
            if len(non_edges) > 0:
                for u, v in random.sample(non_edges,  min(len(non_edges), edge_size-graph.number_of_edges())):
                    graph.add_edge(u, v)
        elif edge_size-graph.number_of_edges()<0:
            edges = list(graph.edges)
            if len(edges) > 0:
                for u, v in random.sample(edges, min(len(edges), graph.number_of_edges()-edge_size)):
                    graph.remove_edge(u,v)
        return graph
    def generate(self, size=None, edge_size=None):
        size = self._get_size(size)
        edge_size = self._get_edge_size(edge_size)
        
        graph = self._generate_graph(size)
        if not edge_size is None:
            graph = self.modify_edges(graph, edge_size)

        feat = self._generate_feat(size)
        
        feat_dict = {i:feat[i] for i in range(size)}
        nx.set_node_attributes(graph, feat_dict, name = "x")
        
        return graph

class ERGenerator(GraphGenerator):
    def __init__(self, sizes, p_alpha=1.3, **kwargs):
        super(ERGenerator, self).__init__(sizes, **kwargs)
        self.p_alpha = p_alpha
        self.kwargs = kwargs

    def _generate_graph(self, size):
        num_nodes = size
        # p follows beta distribution with mean = log2(num_graphs) / num_graphs
        alpha = self.p_alpha
        mean = np.log2(num_nodes) / num_nodes
        
        if "mean" in self.kwargs:
            mean = self.kwargs["mean"]

        beta = alpha / mean - alpha
        p = np.random.beta(alpha, beta)
        graph = nx.gnp_random_graph(num_nodes, p)

        while not nx.is_connected(graph):
            p = np.random.beta(alpha, beta)
            if "p" in self.kwargs:
                p = self.kwargs["p"]
            graph = nx.gnp_random_graph(num_nodes, p)
        #logging.debug('Generated {}-node E-R graphs with average p: {}'.format(
        #       num_nodes, mean))
        return graph

class WSGenerator(GraphGenerator):
    def __init__(self, sizes, density_alpha=1.3, 
            rewire_alpha=2, rewire_beta=2, **kwargs):
        super(WSGenerator, self).__init__(sizes, **kwargs)
        self.density_alpha = density_alpha
        self.rewire_alpha = rewire_alpha
        self.rewire_beta = rewire_beta
        self.kwargs = kwargs

    def _generate_graph(self, size=None):
        num_nodes = self._get_size(size)
        curr_num_graphs = 0

        density_alpha = self.density_alpha
        density_mean = np.log2(num_nodes) / num_nodes
        if "mean" in self.kwargs:
            density_mean = self.kwargs["mean"]
        density_beta = density_alpha / density_mean - density_alpha

        rewire_alpha = self.rewire_alpha
        rewire_beta = self.rewire_beta
        while curr_num_graphs < 1:
            k = int(np.random.beta(density_alpha, density_beta) * num_nodes)
            if "p" in self.kwargs:
                k = int(self.kwargs["p"] * num_nodes)
            k = max(k, 2)
            p = np.random.beta(rewire_alpha, rewire_beta)
            try:
                graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
                curr_num_graphs += 1
            except:
                pass
        logging.debug('Generated {}-node W-S graph with average density: {}'.format(
                num_nodes, density_mean))
        return graph


class SubgraphGenerator(Generator):
    def __init__(self, sizes, size_prob=None, feat_type="random", **kwargs):
        super().__init__(sizes, size_prob)
        self.sub_feat_type = feat_type
    
    def generate(self, graph, size=None, **kwargs):
        raise NotImplementedError

    def _generate_feat(self, graph, size, **kwargs):
        feat_type = self.sub_feat_type
        if "feat_type" in kwargs:
            feat_type = kwargs["feat_type"]

        feat_dim = 0
        node = random.choice(graph.nodes)
        if "x" in node:
            if not isinstance(node["x"], list) and not isinstance(node["x"], np.ndarray):
                for n in graph.nodes:
                    graph.nodes[n]["x"] = [graph.nodes[n]["x"]]
            feat_dim = len(node["x"])

        if feat_type == None:
            return np.zeros([size,feat_dim]).astype(np.float32)
        elif feat_type == "random":
            return np.random.randn(size,feat_dim).astype(np.float32)
        elif feat_type == "ones":
            return np.ones([size,feat_dim]).astype(np.float32)
        elif feat_type == "choice":
            origin_feat = np.array(list(map(lambda x:x[1], list(graph.nodes(data="x")))))
            feat_idx = np.random.choice(len(origin_feat), size)
            feat = origin_feat[feat_idx].astype(np.float32)
            return feat

        return np.zeros(size,feat_dim).astype(np.float32)



class SubTreeGenerator(SubgraphGenerator):
    def __init__(self, sizes, size_prob=None, weight_map=None, **kwargs):
        super(SubTreeGenerator, self).__init__(sizes, size_prob, **kwargs)
        self.weight_map = weight_map
    def generate(self, graph, size=None, weight_map=None, **kwargs):
        size = self._get_size(size)

        if size < 1:
            size = np.round(len(graph.nodes)*size).astype(int)
        
        if weight_map is None:
            weight_map = self.weight_map
        
        node = random.choice(range(len(graph.nodes)))
        neigh = self.subgraph_tree(graph, node, size, weight_map)
        subgraph = nx.subgraph(graph,neigh)
        subgraph = nx.convert_node_labels_to_integers(subgraph)
        return subgraph
    
    def subgraph_tree(self, graph, node, size, weight_map=None):
        start_node = node
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            if not weight_map is None:
                p = list(map(lambda x:weight_map(graph.nodes[x]), frontier))
                p /= sum(p)
            else:
                p = None
            new_node = np.random.choice(list(frontier),p=p)
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return neigh
        else:
            print("No big enough subgraph tree")
            return neigh

class SubERGenerator(SubgraphGenerator, ERGenerator):
    def __init__(self, sizes, p_alpha=1.3, feat_type = "random", **kwargs):
        SubgraphGenerator.__init__(self, sizes, feat_type = feat_type, **kwargs)
        ERGenerator.__init__(self, sizes, p_alpha, **kwargs)
        self.max_time = 10
        if "max_time" in kwargs:
            self.max_time = kwargs["max_time"]

    def _generate_subgraph(self, size, edge_size=None):
        subgraph = self._generate_graph(size)
        if not edge_size is None:
            subgraph = self.modify_edges(subgraph, edge_size)
        return subgraph

    def generate(self, graph, size=None, edge_size=None, iso_check = True, **kwargs):
        size = self._get_size(size)
        if size < 1:
            size = np.round(len(graph.nodes)*size).astype(int)

        edge_size = self._get_edge_size(edge_size)
        if not edge_size is None and edge_size < 1:
            edge_size = np.round(graph.number_of_edges() * size / graph.number_of_nodes() * edge_size).astype(int)
        
        if not iso_check:
            subgraph = self._generate_subgraph(size, edge_size)
            feat = self._generate_feat(graph, size)
            feat_dict = {i:feat[i] for i in range(size)}
            nx.set_node_attributes(subgraph, feat_dict, name = "x")
            return subgraph

        time = 0
        while iso_check and time<=self.max_time:
            subgraph = self._generate_subgraph(size, edge_size)
            feat = self._generate_feat(graph, size)
            feat_dict = {i:feat[i] for i in range(size)}
            nx.set_node_attributes(subgraph, feat_dict, name = "x")

            #isomorphism test with node feature
            nx.set_node_attributes(subgraph, {i:hash(str(list(subgraph.nodes[i]["x"]))) for i in range(subgraph.number_of_nodes())}, name = "x_hash")
            nx.set_node_attributes(graph, {i:hash(str(list(graph.nodes[i]["x"]))) for i in range(graph.number_of_nodes())}, name = "x_hash")
            GM = isomorphism.GraphMatcher(graph, subgraph, node_match=isomorphism.categorical_node_match("x_hash", 0))

            iso_check = limit_time(GM.subgraph_is_isomorphic, MAX_MATCHING_TIME, 1)
            if iso_check is None:
                print("Exact check out of time")
                iso_check = False
            time += 1

        if time > self.max_time:
            return None
        #else:
        #    print("iso_check:", iso_check)
        return subgraph

class SubNegTreeGenerator(SubTreeGenerator):
    def __init__(self, sizes, size_prob=None, **kwargs):
        SubTreeGenerator.__init__(self, sizes, size_prob, **kwargs)
        self.max_time = 20
        if "max_time" in kwargs:
            self.max_time = kwargs["max_time"]

    def generate(self, graph, size=None, perturb_size=5, edge_size=None, iso_check = True, input_subgraph=None, **kwargs):
        size = self._get_size(size)
        if size < 1:
            size = np.round(len(graph.nodes)*size).astype(int)

        edge_size = self._get_edge_size(edge_size)
        if not edge_size is None and edge_size < 1:
            edge_size = np.round(graph.number_of_edges() * size / graph.number_of_nodes() * edge_size).astype(int)
        
        if not iso_check:
            subgraph = self._generate_subgraph(graph, size, perturb_size, edge_size, input_subgraph)
            return subgraph

        time = 0
        while iso_check and time<=self.max_time:
            subgraph = self._generate_subgraph(graph, size, perturb_size, edge_size, input_subgraph)

            #isomorphism test with node feature
            nx.set_node_attributes(subgraph, {i:hash(str(list(subgraph.nodes[i]["x"]))) for i in range(subgraph.number_of_nodes())}, name = "x_hash")
            nx.set_node_attributes(graph, {i:hash(str(list(graph.nodes[i]["x"]))) for i in range(graph.number_of_nodes())}, name = "x_hash")
            GM = isomorphism.GraphMatcher(graph, subgraph, node_match=isomorphism.categorical_node_match("x_hash", 0))

            iso_check = limit_time(GM.subgraph_is_isomorphic, MAX_MATCHING_TIME, 1)
            if iso_check is None:
                print("Exact check out of time")
                iso_check = False
            time += 1

        if time > self.max_time:
            return None

        return subgraph

    def _generate_subgraph(self,graph, size, perturb_size, edge_size=None, input_subgraph=None):

        if input_subgraph is None:
            node = random.choice(range(len(graph.nodes)))
            neigh = self.subgraph_tree(graph, node, size)
            subgraph = nx.subgraph(graph,neigh)
            subgraph = nx.convert_node_labels_to_integers(subgraph)
        else:
            subgraph = copy.deepcopy(input_subgraph)

        edges = list(subgraph.edges)
        non_edges = list(nx.non_edges(subgraph))

        if perturb_size<1:
            perturb_size = int(np.ceil(perturb_size*len(edges)))
        
        perturb_size =  min(len(edges),len(non_edges), perturb_size)

        if len(edges) > 0:
            for u, v in random.sample(edges, perturb_size):
                subgraph.remove_edge(u, v)

            for u, v in random.sample(non_edges, perturb_size):
                subgraph.add_edge(u, v)

        if not edge_size is None:
            subgraph = self.modify_edges(subgraph, edge_size)

        return subgraph
