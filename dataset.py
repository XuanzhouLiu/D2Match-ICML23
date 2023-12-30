import random
import numpy as np
from itertools import cycle
from pathlib import Path
from sklearn.utils import shuffle
import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as pyg_utils
from torch_geometric.datasets import TUDataset, PPI, QM9, GNNBenchmarkDataset

from ogb.graphproppred import PygGraphPropPredDataset

from generator import ERGenerator, SubNegTreeGenerator, WSGenerator, SubTreeGenerator, SubERGenerator
import networkx as nx

class PairData(Data):
    def __init__(self, data_s = None, data_t = None):
        super().__init__()
        self.t_nodes = 0
        self.s_nodes = 0
        if data_s is None:
            self.edge_index_s = torch.tensor([])
            self.x_s = torch.tensor([])
        else:
            self.edge_index_s = data_s.edge_index
            self.x_s = data_s.x
            if not data_s.x is None:
                self.s_nodes += len(data_s.x)
        
        if data_t is None:
            self.edge_index_t = None
            self.x_t = None
        else:
            self.edge_index_t = data_t.edge_index
            self.x_t = data_t.x
            if not data_s.x is None:
                self.t_nodes += len(data_t.x)
        
        self.num_nodes = self.s_nodes+self.t_nodes
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def target_data(self):
        data = Data(x = self.x_t, edge_index = self.edge_index_t)
        if hasattr(self, "x_t_batch"):
            data.batch = self.x_t_batch
        return data

    def subgraph_data(self):
        data = Data(x = self.x_s, edge_index = self.edge_index_s)
        if hasattr(self, "x_s_batch"):
            data.batch = self.x_s_batch
        return data

    def __cat_dim__(self, key, value, *args, **kwargs):
         if key == 'degree_assign':
             return None
         else:
             return super().__cat_dim__(key, value, *args, **kwargs)

class RealDataset(InMemoryDataset):
    def __init__(self, root, data_name,size = None, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 5, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True,transform = None, pre_transform=None, **kwargs):

        self.kwargs = kwargs
        self.data_name = data_name
        self.neg_feat_type = neg_feat_type

        self.size = size
        
        if not isinstance(subgraph_sizes, list):
            self.subgraph_sizes = [subgraph_sizes]
        else:
            self.subgraph_sizes = subgraph_sizes

        if not isinstance(subgraph_edge_sizes, list) and not subgraph_edge_sizes is None:
            self.subgraph_edge_sizes = [subgraph_edge_sizes]
        else:
            self.subgraph_edge_sizes = subgraph_edge_sizes

        self.positive = positive
        if not isinstance(pos_subgraph_type, list):
            self.pos_subgraph_type_list = [pos_subgraph_type]
        else:
            self.pos_subgraph_type_list = pos_subgraph_type

        if not isinstance(neg_subgraph_type, list):
            self.neg_subgraph_type_list = [neg_subgraph_type]
        else:
            self.neg_subgraph_type_list = neg_subgraph_type

        self.iso_check = iso_check
        self._root = Path(root) / self._file_path()
        if not self._root.exists():
            self._root.mkdir()
        self.p_transform = pre_transform

        super().__init__(self._root)

        if self.p_transform:
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

        if not transform is None:
            self.data = transform(self.data)

    def _dataset(self):
        if self.data_name == "enzymes":
            dataset = TUDataset(root = self._root / self.data_name, name="ENZYMES")
        elif self.data_name == "proteins":
            dataset = TUDataset(root = self._root / self.data_name, name="PROTEINS")
        elif self.data_name == "cox2":
            dataset = TUDataset(root = self._root / self.data_name, name="COX2")
        elif self.data_name == "aids":
            dataset = TUDataset(root= self._root / self.data_name, name="AIDS")
        elif self.data_name == "reddit-binary":
            dataset = TUDataset(root= self._root / self.data_name, name="REDDIT-BINARY")
        elif self.data_name == "imdb-binary":
            dataset = TUDataset(root= self._root / self.data_name, name="IMDB-BINARY")
        elif self.data_name == "firstmm_db":
            dataset = TUDataset(root= self._root / self.data_name, name="FIRSTMM_DB")
        elif self.data_name == "mutag":
            dataset = TUDataset(root= self._root / self.data_name, name="MUTAG")
        elif self.data_name == "dblp":
            dataset = TUDataset(root= self._root / self.data_name, name="DBLP_v1")
        elif self.data_name == "ppi":
            dataset = PPI(root= self._root / self.data_name)
        elif self.data_name == "qm9":
            dataset = QM9(root= self._root / self.data_name)
        elif self.data_name == "cifar10":
            dataset = GNNBenchmarkDataset(root= self._root / self.data_name, name="CIFAR10")
        elif self.data_name == "mnist":
            dataset = GNNBenchmarkDataset(root= self._root / self.data_name, name="MNIST")
        elif self.data_name == "ogbg-molhiv":
            dataset = PygGraphPropPredDataset(root= self._root / self.data_name, name = "ogbg-molhiv")
        elif self.data_name == "ogbg-molpcba":
            dataset = PygGraphPropPredDataset(root= self._root / self.data_name, name = "ogbg-molpcba")
        return dataset
        #elif self.data_name == "atlas":
        #    dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]

    @property
    def raw_file_names(self):
        return [self.data_name]

    @property
    def processed_file_names(self):
        return ['data.pt'] if self.p_transform is None else ['data.pt', 'pretrans_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        self._dataset()

    def _file_path(self):
        if "name" in self.kwargs:
            name = self.kwargs["name"]
        else:
            name = "{}-{}{}{}".format(self.data_name, \
            "_".join(self.pos_subgraph_type_list), "_".join(self.neg_subgraph_type_list), "_".join([str(i) for i in self.subgraph_sizes]))
        return str(name) 

    def num_classes(self):
        return 2

    def process(self):
        print("Begin Generating Data")
        self.create_generator()

        dataset = self._dataset()

        size = len(dataset)
        if not self.size is None:
            size = self.size

        positive = self.positive
        if self.positive < 1:
            positive = np.round(size * self.positive).astype(int)

        negative = size - positive
        data_list, num_edges, num_sub_edges = self.generate_positive(positive, dataset)
        data_list.extend(self.generate_negative(negative, num_edges, num_sub_edges, dataset))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        pt_data_list = []
        if not self.p_transform is None:
            print("Pretransform...")
            for i in tqdm.tqdm(range(len(data_list))):
                pair = self.p_transform(data_list[i])
                pair.y = data_list[i].y
                pt_data_list.append(pair)
            data, slices = self.collate(pt_data_list)
            torch.save((data, slices), self.processed_paths[1])
        print("Done")

    def generate_positive(self, size, dataset):
        print("Generating Positive Samples")
        num_sub_edges = []
        num_edges = []

        data_list = []
        graph_iter = cycle(dataset[np.random.permutation(np.arange(len(dataset)))])
        for j in tqdm.tqdm(range(size)):
            sub_generator = random.choice(self.pos_subgraph_generator_list)
            graph_data = next(graph_iter)
            # some dataset like reddit may contain no feature
            if graph_data.x is None:
                graph_data = Data(torch.ones(graph_data.num_nodes, 4).float(), graph_data.edge_index)

            graph = pyg_utils.to_networkx(graph_data, ["x"])
            subgraph = sub_generator.generate(graph)

            num_edges.append(graph.number_of_edges())
            num_sub_edges.append(subgraph.number_of_edges())

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 1
            data_list.append(pair)
        return data_list, num_edges, num_sub_edges

    def generate_negative(self, size, num_edges, num_sub_edges, dataset):
        print("Generating Positive Samples")

        sub_std = np.std(num_sub_edges)
        sub_mean = np.mean(num_sub_edges)
        data_list = []
        graph_iter = cycle(dataset[np.random.permutation(np.arange(len(dataset)))])
        for j in tqdm.tqdm(range(size)):
            sub_generator = random.choice(self.neg_subgraph_generator_list)

            sub_edge_size = None

            if j < len(num_sub_edges):
                sub_edge_size = num_sub_edges[j]
            else:
                sub_edge_size = int(np.round(np.random.normal(sub_mean, sub_std)))

            graph_data = next(graph_iter)
            # some dataset like reddit may contain no feature
            if graph_data.x is None:
                graph_data = Data(torch.ones(graph_data.num_nodes, 4).float(), graph_data.edge_index)
                
            graph = pyg_utils.to_networkx(graph_data, ["x"])
            subgraph = sub_generator.generate(graph, edge_size=sub_edge_size, iso_check=self.iso_check)

            t = 0
            while subgraph is None and t<10:
                print("Generate Again")
                subgraph = sub_generator.generate(graph,edge_size=sub_edge_size, iso_check=self.iso_check)
                t+=1
            if subgraph is None:
                subgraph = sub_generator.generate(graph,edge_size=sub_edge_size, iso_check=False)

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 0
            data_list.append(pair)
        return data_list

    def create_generator(self):
        self.pos_subgraph_generator_list = []
        for subgraph_type in self.pos_subgraph_type_list:
            if subgraph_type == "subtree":
                self.pos_subgraph_generator_list.append(SubTreeGenerator(self.subgraph_sizes, **self.kwargs))
            else:
                raise RuntimeError("Positive Subgraph Type {} Not Included".format(subgraph_type))

        self.neg_subgraph_generator_list = []
        for subgraph_type in self.neg_subgraph_type_list:
            if subgraph_type == "ER":
                self.neg_subgraph_generator_list.append(SubERGenerator(self.subgraph_sizes, edge_sizes = self.subgraph_edge_sizes, feat_type = self.neg_feat_type, **self.kwargs))
            else:
                raise RuntimeError("Negative Subgraph Type {} Not Included".format(subgraph_type))



class HardDataset(InMemoryDataset):
    def __init__(self, root, data_name,size = None, positive = 0.5, same_pn_graph=True,\
        pos_subgraph_type = "subtree", neg_subgraph_perturb = 0.2, subgraph_sizes = 5, subgraph_edge_sizes = None, \
        iso_check = True, transform = None, pre_transform=None, **kwargs):

        self.kwargs = kwargs
        self.data_name = data_name

        self.size = size
        
        if not isinstance(subgraph_sizes, list):
            self.subgraph_sizes = [subgraph_sizes]
        else:
            self.subgraph_sizes = subgraph_sizes

        if not isinstance(subgraph_edge_sizes, list) and not subgraph_edge_sizes is None:
            self.subgraph_edge_sizes = [subgraph_edge_sizes]
        else:
            self.subgraph_edge_sizes = subgraph_edge_sizes

        self.positive = positive
        if not isinstance(pos_subgraph_type, list):
            self.pos_subgraph_type_list = [pos_subgraph_type]
        else:
            self.pos_subgraph_type_list = pos_subgraph_type

        self.same_pn_graph = same_pn_graph
        self.neg_subgraph_perturb = neg_subgraph_perturb

        self.iso_check = iso_check
        self._root = Path(root) / self._file_path()
        if not self._root.exists():
            self._root.mkdir()
        self.p_transform = pre_transform

        super().__init__(self._root)

        if self.p_transform:
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

        if not transform is None:
            self.data = transform(self.data)

    def _dataset(self):
        if self.data_name == "enzymes":
            dataset = TUDataset(root = self._root / self.data_name, name="ENZYMES")
        elif self.data_name == "proteins":
            dataset = TUDataset(root = self._root / self.data_name, name="PROTEINS")
        elif self.data_name == "cox2":
            dataset = TUDataset(root = self._root / self.data_name, name="COX2")
        elif self.data_name == "aids":
            dataset = TUDataset(root= self._root / self.data_name, name="AIDS")
        elif self.data_name == "reddit-binary":
            dataset = TUDataset(root= self._root / self.data_name, name="REDDIT-BINARY")
        elif self.data_name == "imdb-binary":
            dataset = TUDataset(root= self._root / self.data_name, name="IMDB-BINARY")
        elif self.data_name == "firstmm_db":
            dataset = TUDataset(root= self._root / self.data_name, name="FIRSTMM_DB")
        elif self.data_name == "mutag":
            dataset = TUDataset(root= self._root / self.data_name, name="MUTAG")
        elif self.data_name == "dblp":
            dataset = TUDataset(root= self._root / self.data_name, name="DBLP_v1")
        elif self.data_name == "ppi":
            dataset = PPI(root= self._root / self.data_name)
        elif self.data_name == "qm9":
            dataset = QM9(root= self._root / self.data_name)
        elif self.data_name == "cifar10":
            dataset = GNNBenchmarkDataset(root= self._root / self.data_name, name="CIFAR10")
        elif self.data_name == "mnist":
            dataset = GNNBenchmarkDataset(root= self._root / self.data_name, name="MNIST")
        elif self.data_name == "ogbg-molhiv":
            dataset = PygGraphPropPredDataset(root= self._root / self.data_name, name = "ogbg-molhiv")
        elif self.data_name == "ogbg-molpcba":
            dataset = PygGraphPropPredDataset(root= self._root / self.data_name, name = "ogbg-molpcba")
        return dataset
        #elif self.data_name == "atlas":
        #    dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]

    @property
    def raw_file_names(self):
        return [self.data_name]

    @property
    def processed_file_names(self):
        return ['data.pt'] if self.p_transform is None else ['data.pt', 'pretrans_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        self._dataset()

    def _file_path(self):
        if "name" in self.kwargs:
            name = self.kwargs["name"]
        else:
            name = "Hard{}-{}{}".format(self.data_name, \
            "_".join(self.pos_subgraph_type_list), "_".join([str(i) for i in self.subgraph_sizes]))
        return str(name) 

    def num_classes(self):
        return 2

    def process(self):
        print("Begin Generating Data")
        dataset = self._dataset()
        self.create_generator(dataset)

        size = len(dataset)
        if not self.size is None:
            size = self.size

        positive = self.positive
        if self.positive < 1:
            positive = np.round(size * self.positive).astype(int)

        negative = size - positive
        data_list, graph_list, subgraph_list, num_edges, num_sub_edges = self.generate_positive(positive, dataset)
        data_list.extend(self.generate_negative(negative, graph_list, subgraph_list, num_edges, num_sub_edges, dataset))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        pt_data_list = []
        if not self.p_transform is None:
            print("Pretransform...")
            for i in tqdm.tqdm(range(len(data_list))):
                pair = self.p_transform(data_list[i])
                pair.y = data_list[i].y
                pt_data_list.append(pair)
            data, slices = self.collate(pt_data_list)
            torch.save((data, slices), self.processed_paths[1])
        print("Done")

    def generate_positive(self, size, dataset):
        print("Generating Positive Samples")

        data_list = []
        graph_list = []
        subgraph_list = []
        num_edges = []
        num_sub_edges = []

        graph_iter = cycle(dataset[np.random.permutation(np.arange(len(dataset)))])
        for j in tqdm.tqdm(range(size)):
            sub_generator = random.choice(self.pos_subgraph_generator_list)
            graph_data = next(graph_iter)
            # some dataset like reddit may contain no feature
            if graph_data.x is None:
                graph_data = Data(torch.ones(graph_data.num_nodes, 4).float(), graph_data.edge_index)

            graph = pyg_utils.to_networkx(graph_data, ["x"])
            subgraph = sub_generator.generate(graph)

            graph_list.append(graph)
            subgraph_list.append(subgraph)
            num_edges.append(graph.number_of_edges())
            num_sub_edges.append(subgraph.number_of_edges())

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 1
            data_list.append(pair)
        return data_list, graph_list, subgraph_list, num_edges, num_sub_edges

    def generate_negative(self, size, graph_list, subgraph_list, num_edges, num_sub_edges, dataset):
        print("Generating Negative Samples")
        
        sub_std = np.std(num_sub_edges)
        sub_mean = np.mean(num_sub_edges)
        graph_iter = cycle(dataset[np.random.permutation(np.arange(len(dataset)))])
        data_list = []
        for j in tqdm.tqdm(range(size)):
            sub_generator = self.neg_subgraph_generator_list[0]
            
            sub_edge_size =None
            input_subgraph = None
            if self.same_pn_graph:
                if j < len(graph_list):
                    graph = graph_list[j]
                    input_subgraph = subgraph_list[j]
                else:
                    graph_idx = random.choice(range(len(graph_list)))
                    graph = graph_list[graph_idx]
                    input_subgraph = subgraph_list[graph_idx]
            else:
                if j < len(num_sub_edges):
                    sub_edge_size = num_sub_edges[j]
                else:
                    sub_edge_size = int(np.round(np.random.normal(sub_mean, sub_std)))

                graph_data = next(graph_iter)
                # some dataset like reddit may contain no feature
                if graph_data.x is None:
                    graph_data = Data(torch.ones(graph_data.num_nodes, 4).float(), graph_data.edge_index)
                graph = pyg_utils.to_networkx(graph_data, ["x"])

            
            subgraph = sub_generator.generate(graph, perturb_size=self.neg_subgraph_perturb, edge_size=sub_edge_size, iso_check=self.iso_check, input_subgraph=input_subgraph)

            t = 0
            while subgraph is None and t<10:
                print("Generate Again")
                subgraph = sub_generator.generate(graph, perturb_size=self.neg_subgraph_perturb, edge_size=sub_edge_size, iso_check=self.iso_check, input_subgraph=input_subgraph)
                t+=1
            if subgraph is None:
                subgraph = sub_generator.generate(graph, perturb_size=self.neg_subgraph_perturb, edge_size=sub_edge_size, iso_check=False, input_subgraph=input_subgraph)

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 0
            data_list.append(pair)
        return data_list

    def create_generator(self, dataset):
        self.pos_subgraph_generator_list = []
        for subgraph_type in self.pos_subgraph_type_list:
            if subgraph_type == "subtree":
                self.pos_subgraph_generator_list.append(SubTreeGenerator(self.subgraph_sizes, **self.kwargs))
            else:
                raise RuntimeError("Positive Subgraph Type {} Not Included".format(subgraph_type))

        self.neg_subgraph_generator_list = [SubNegTreeGenerator(self.subgraph_sizes, **self.kwargs)]



class HardInbalancedDataset(InMemoryDataset):
    def __init__(self, root, data_name,size = None, positive = 0.5, same_pn_graph=True,\
        pos_subgraph_type = "subtree", neg_subgraph_perturb = 0.2, subgraph_sizes = 5, subgraph_edge_sizes = None, \
        iso_check = True, transform = None, pre_transform=None, **kwargs):

        self.kwargs = kwargs
        self.data_name = data_name

        self.size = size
        
        if not isinstance(subgraph_sizes, list):
            self.subgraph_sizes = [subgraph_sizes]
        else:
            self.subgraph_sizes = subgraph_sizes

        if not isinstance(subgraph_edge_sizes, list) and not subgraph_edge_sizes is None:
            self.subgraph_edge_sizes = [subgraph_edge_sizes]
        else:
            self.subgraph_edge_sizes = subgraph_edge_sizes

        self.positive = positive
        if not isinstance(pos_subgraph_type, list):
            self.pos_subgraph_type_list = [pos_subgraph_type]
        else:
            self.pos_subgraph_type_list = pos_subgraph_type

        self.same_pn_graph = same_pn_graph
        self.neg_subgraph_perturb = neg_subgraph_perturb

        self.iso_check = iso_check
        self._root = Path(root) / self._file_path()
        if not self._root.exists():
            self._root.mkdir()
        self.p_transform = pre_transform

        super().__init__(self._root)

        if self.p_transform:
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

        if not transform is None:
            self.data = transform(self.data)

    def _dataset(self):
        if self.data_name == "enzymes":
            dataset = TUDataset(root = self._root / self.data_name, name="ENZYMES")
        elif self.data_name == "proteins":
            dataset = TUDataset(root = self._root / self.data_name, name="PROTEINS")
        elif self.data_name == "cox2":
            dataset = TUDataset(root = self._root / self.data_name, name="COX2")
        elif self.data_name == "aids":
            dataset = TUDataset(root= self._root / self.data_name, name="AIDS")
        elif self.data_name == "reddit-binary":
            dataset = TUDataset(root= self._root / self.data_name, name="REDDIT-BINARY")
        elif self.data_name == "imdb-binary":
            dataset = TUDataset(root= self._root / self.data_name, name="IMDB-BINARY")
        elif self.data_name == "firstmm_db":
            dataset = TUDataset(root= self._root / self.data_name, name="FIRSTMM_DB")
        elif self.data_name == "mutag":
            dataset = TUDataset(root= self._root / self.data_name, name="MUTAG")
        elif self.data_name == "dblp":
            dataset = TUDataset(root= self._root / self.data_name, name="DBLP_v1")
        elif self.data_name == "ppi":
            dataset = PPI(root= self._root / self.data_name)
        elif self.data_name == "qm9":
            dataset = QM9(root= self._root / self.data_name)
        elif self.data_name == "cifar10":
            dataset = GNNBenchmarkDataset(root= self._root / self.data_name, name="CIFAR10")
        elif self.data_name == "mnist":
            dataset = GNNBenchmarkDataset(root= self._root / self.data_name, name="MNIST")
        elif self.data_name == "ogbg-molhiv":
            dataset = PygGraphPropPredDataset(root= self._root / self.data_name, name = "ogbg-molhiv")
        elif self.data_name == "ogbg-molpcba":
            dataset = PygGraphPropPredDataset(root= self._root / self.data_name, name = "ogbg-molpcba")
        return dataset
        #elif self.data_name == "atlas":
        #    dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]

    @property
    def raw_file_names(self):
        return [self.data_name]

    @property
    def processed_file_names(self):
        return ['data.pt'] if self.p_transform is None else ['data.pt', 'pretrans_data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        self._dataset()

    def _file_path(self):
        if "name" in self.kwargs:
            name = self.kwargs["name"]
        else:
            name = "Hard{}-{}{}".format(self.data_name, \
            "_".join(self.pos_subgraph_type_list), "_".join([str(i) for i in self.subgraph_sizes]))
        return str(name) 

    def num_classes(self):
        return 2

    def process(self):
        print("Begin Generating Data")
    
        dataset = self._dataset()
        self.create_generator(dataset)

        size = len(dataset)
        if not self.size is None:
            size = self.size

        positive = self.positive
        if self.positive < 1:
            positive = np.round(size * self.positive).astype(int)

        negative = size - positive
        data_list, graph_list, subgraph_list, num_edges, num_sub_edges = self.generate_positive(positive, dataset)
        data_list.extend(self.generate_negative(negative, graph_list, subgraph_list, num_edges, num_sub_edges, dataset))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        pt_data_list = []
        if not self.p_transform is None:
            print("Pretransform...")
            for i in tqdm.tqdm(range(len(data_list))):
                pair = self.p_transform(data_list[i])
                pair.y = data_list[i].y
                pt_data_list.append(pair)
            data, slices = self.collate(pt_data_list)
            torch.save((data, slices), self.processed_paths[1])
        print("Done")

    def generate_positive(self, size, dataset):
        print("Generating Positive Samples")

        data_list = []
        graph_list = []
        subgraph_list = []
        num_edges = []
        num_sub_edges = []

        graph_iter = cycle(dataset[np.random.permutation(np.arange(len(dataset)))])
        for j in tqdm.tqdm(range(size)):
            sub_generator = random.choice(self.pos_subgraph_generator_list)
            graph_data = next(graph_iter)
            # some dataset like reddit may contain no feature
            if graph_data.x is None:
                graph_data = Data(torch.ones(graph_data.num_nodes, 4).float(), graph_data.edge_index)

            graph = pyg_utils.to_networkx(graph_data, ["x"])
            subgraph = sub_generator.generate(graph)

            graph_list.append(graph)
            subgraph_list.append(subgraph)
            num_edges.append(graph.number_of_edges())
            num_sub_edges.append(subgraph.number_of_edges())

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 1
            data_list.append(pair)
        return data_list, graph_list, subgraph_list, num_edges, num_sub_edges

    def generate_negative(self, size, graph_list, subgraph_list, num_edges, num_sub_edges, dataset):
        print("Generating Negative Samples")
        
        sub_std = np.std(num_sub_edges)
        sub_mean = np.mean(num_sub_edges)
        graph_iter = cycle(dataset[np.random.permutation(np.arange(len(dataset)))])
        data_list = []
        for j in tqdm.tqdm(range(size)):
            sub_generator = self.neg_subgraph_generator_list[0]
            
            sub_edge_size =None
            input_subgraph = None
            if self.same_pn_graph:
                if j < len(graph_list):
                    graph = graph_list[j]
                    input_subgraph = subgraph_list[j]
                else:
                    graph_idx = random.choice(range(len(graph_list)))
                    graph = graph_list[graph_idx]
                    input_subgraph = subgraph_list[graph_idx]
            else:
                if j < len(num_sub_edges):
                    sub_edge_size = num_sub_edges[j]
                else:
                    sub_edge_size = int(np.round(np.random.normal(sub_mean, sub_std)))

                graph_data = next(graph_iter)
                # some dataset like reddit may contain no feature
                if graph_data.x is None:
                    graph_data = Data(torch.ones(graph_data.num_nodes, 4).float(), graph_data.edge_index)
                graph = pyg_utils.to_networkx(graph_data, ["x"])

            
            subgraph = sub_generator.generate(graph, perturb_size=self.neg_subgraph_perturb, edge_size=sub_edge_size, iso_check=self.iso_check, input_subgraph=input_subgraph)

            t = 0
            while subgraph is None and t<10:
                print("Generate Again")
                subgraph = sub_generator.generate(graph, perturb_size=self.neg_subgraph_perturb, edge_size=sub_edge_size, iso_check=self.iso_check, input_subgraph=input_subgraph)
                t+=1
            if subgraph is None:
                subgraph = sub_generator.generate(graph, perturb_size=self.neg_subgraph_perturb, edge_size=sub_edge_size, iso_check=False, input_subgraph=input_subgraph)

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 0
            data_list.append(pair)
        return data_list

    def create_generator(self, dataset):
        self.pos_subgraph_generator_list = []
        for subgraph_type in self.pos_subgraph_type_list:
            if subgraph_type == "subtree":
                hist = torch.sum(dataset.data.x, dim=0)
                p = (1/hist)**2
                p[p==float('inf')] = 0
                p = p.numpy()
                def weight_map(node):
                    return np.sum(np.array(node["x"])*p)

                self.pos_subgraph_generator_list.append(SubTreeGenerator(self.subgraph_sizes, weight_map=weight_map, **self.kwargs))
            else:
                raise RuntimeError("Positive Subgraph Type {} Not Included".format(subgraph_type))

        self.neg_subgraph_generator_list = [SubNegTreeGenerator(self.subgraph_sizes, **self.kwargs)]




class SynDataset(InMemoryDataset):
    def __init__(self, root, size = 100, positive = 50, \
        graph_type = "ER", graph_sizes = 20, graph_edge_sizes=None,\
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 5, subgraph_edge_sizes=None, fix_edge_size=True, \
        feat_type = "random", feat_dim = 1, iso_check = True, transform = None, pre_transform = None, **kwargs):
        self.kwargs = kwargs

        self.size = size

        if not isinstance(graph_type, list):
            self.graph_type_list = [graph_type]
        else:
            self.graph_type_list = graph_type
        
        if not isinstance(graph_sizes, list):
            self.graph_sizes = [graph_sizes]
        else:
            self.graph_sizes = graph_sizes

        if not isinstance(graph_edge_sizes, list) and not graph_edge_sizes is None:
            self.graph_edge_sizes = [graph_edge_sizes]
        else:
            self.graph_edge_sizes = graph_edge_sizes

        if not isinstance(subgraph_edge_sizes, list) and not subgraph_edge_sizes is None:
            self.subgraph_edge_sizes = [subgraph_edge_sizes]
        else:
            self.subgraph_edge_sizes = subgraph_edge_sizes

        self.positive = positive
        if not isinstance(pos_subgraph_type, list):
            self.pos_subgraph_type_list = [pos_subgraph_type]
        else:
            self.pos_subgraph_type_list = pos_subgraph_type

        if not isinstance(neg_subgraph_type, list):
            self.neg_subgraph_type_list = [neg_subgraph_type]
        else:
            self.neg_subgraph_type_list = neg_subgraph_type

        if not isinstance(subgraph_sizes, list):
            self.subgraph_sizes = [subgraph_sizes]
        else:
            self.subgraph_sizes = subgraph_sizes
        
        self.fix_edge_size = fix_edge_size

        self.feat_type = feat_type
        self.feat_dim = feat_dim

        self.iso_check = iso_check

        self.p_transform = pre_transform

        self._root = Path(root) / self._file_path()
        if not self._root.exists():
            self._root.mkdir()

        super().__init__(self._root)
        if self.p_transform:
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

        if not transform is None:
            self.data = transform(self.data)

    @property
    def raw_file_names(self):
        return [""]

    @property
    def processed_file_names(self):
        return ['data.pt'] if self.p_transform is None else ['data.pt', 'pretrans_data.pt']

    def _file_path(self):
        if "name" in self.kwargs:
            name = self.kwargs["name"]
        else:
            name = "SynData-{}-{}{}-{}{}{}-{}{}.pt".format(self.size, \
            "_".join(self.graph_type_list), "_".join([str(i) for i in self.graph_sizes]), \
            "_".join(self.pos_subgraph_type_list), "_".join(self.neg_subgraph_type_list), "_".join([str(i) for i in self.subgraph_sizes]), \
            str(self.feat_type), self.feat_dim)
        return str(name)

    def num_classes(self):
        return 2

    def process(self):
        print("Begin Generating Data")
        self.create_generator()

        data_list = []
        positive = self.positive
        if self.positive < 1:
            positive = np.round(self.size * self.positive)
        negative = self.size - positive
        
        data_list, num_edges, num_sub_edges = self.generate_positive(positive)
        data_list.extend(self.generate_negative(negative, num_edges, num_sub_edges))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        pt_data_list = []
        if not self.p_transform is None:
            print("Pretransform...")
            for i in tqdm.tqdm(range(len(data_list))):
                pair = self.p_transform(data_list[i])
                pair.y = data_list[i].y
                pt_data_list.append(pair)
            data, slices = self.collate(pt_data_list)
            torch.save((data, slices), self.processed_paths[1])

        print("Done")

    def generate_positive(self, size):
        print("Generating Positive Samples")
        num_sub_edges = []
        num_edges = []
        data_list = []
        for j in tqdm.tqdm(range(size)):
            generator = random.choice(self.generator_list)
            sub_generator = random.choice(self.pos_subgraph_generator_list)

            graph = generator.generate()
            subgraph = sub_generator.generate(graph)

            num_edges.append(graph.number_of_edges())
            num_sub_edges.append(subgraph.number_of_edges())

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 1
            data_list.append(pair)

        return data_list, num_edges, num_sub_edges

    def generate_negative(self, size, num_edges, num_sub_edges):
        data_list = []
        print("Generating Negative Samples")
        sub_std = np.std(num_sub_edges)
        sub_mean = np.mean(num_sub_edges)
        std = np.std(num_edges)
        mean = np.mean(num_edges)

        for j in tqdm.tqdm(range(size)):
            generator = random.choice(self.generator_list)
            sub_generator = random.choice(self.neg_subgraph_generator_list)

            edge_size = None
            sub_edge_size = None
            if self.fix_edge_size:
                if j < len(num_sub_edges):
                    edge_size = num_edges[j]
                    sub_edge_size = num_sub_edges[j]
                else:
                    edge_size = int(np.round(np.random.normal(mean, std)))
                    sub_edge_size = int(np.round(np.random.normal(sub_mean, sub_std)))

            graph = generator.generate(edge_size=edge_size)
            subgraph = sub_generator.generate(graph,edge_size=sub_edge_size, iso_check=self.iso_check)

            t = 0
            while subgraph is None and t<10:
                print("Generate Again")
                graph = generator.generate(edge_size=edge_size)
                subgraph = sub_generator.generate(graph,edge_size=sub_edge_size, iso_check=self.iso_check)
                t+=1
            if subgraph is None:
                graph = generator.generate(edge_size=edge_size)
                subgraph = sub_generator.generate(graph,edge_size=sub_edge_size, iso_check=False)

            data_s = pyg_utils.from_networkx(subgraph, ["x"])
            data_t = pyg_utils.from_networkx(graph, ["x"])
            pair = PairData(data_s,data_t)
            pair.y = 0
            data_list.append(pair)
        return data_list

    def create_generator(self):
        self.generator_list = []
        for graph_type in self.graph_type_list:
            if graph_type == "ER":
                self.generator_list.append(ERGenerator(self.graph_sizes, edge_sizes = self.graph_edge_sizes, feat_type = self.feat_type, feat_dim = self.feat_dim, **self.kwargs))
            elif graph_type == "WS":
                self.generator_list.append(WSGenerator(self.graph_sizes, edge_sizes = self.graph_edge_sizes, feat_type = self.feat_type, feat_dim = self.feat_dim, **self.kwargs))
            else:
                raise RuntimeError("Graph Type {} Not Included".format(graph_type))

        self.pos_subgraph_generator_list = []
        for subgraph_type in self.pos_subgraph_type_list:
            if subgraph_type == "subtree":
                self.pos_subgraph_generator_list.append(SubTreeGenerator(self.subgraph_sizes, **self.kwargs))
            else:
                raise RuntimeError("Positive Subgraph Type {} Not Included".format(subgraph_type))

        self.neg_subgraph_generator_list = []
        for subgraph_type in self.neg_subgraph_type_list:
            if subgraph_type == "ER":
                self.neg_subgraph_generator_list.append(SubERGenerator(self.subgraph_sizes, edge_sizes = self.subgraph_edge_sizes,  feat_type = self.feat_type, **self.kwargs))
            else:
                raise RuntimeError("Negative Subgraph Type {} Not Included".format(subgraph_type))

from sklearn.model_selection import StratifiedKFold
from numpy.random import default_rng
def get_dataset(root, dataset_name, trans=None, pre_trans=None, n_fold=None, fold_idx=0, random_seed=None):
    if dataset_name == "cox2":
        dataset = RealDataset(root, "cox2",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="Cox2-1000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "syn":
        dataset = SynDataset(root, size=1000, positive=500, graph_type=["ER","WS"], graph_sizes=40,\
            pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=15,
            feat_type="ones", feat_dim=10, name = "Syn40_15-1000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "enzymes":
        dataset = RealDataset(root, "enzymes",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="Enzymes-1000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "proteins":
        dataset = RealDataset(root, "proteins",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="Proteins-1000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "mutag":
        dataset = RealDataset(root, "mutag",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="MUTAG-1000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "aids":
        dataset = RealDataset(root, "aids",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="Aids-2000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "firstmm":
        dataset = RealDataset(root, "firstmm_db",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="Firstmm-1000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "imdb":
        dataset = RealDataset(root, "imdb-binary",size = 2000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="Imdb-2000", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "syn_dnf":
        dataset = SynDataset(root, size=1000, positive=500, graph_type=["ER","WS"], graph_sizes=40,\
            pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=15, fix_edge_size=False,
            feat_type="ones", feat_dim=10, name = "Syn40_15-1000DensityNotFixed", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "enzymes_hard":
        dataset = HardDataset(root, "enzymes", size=500, positive=0.5, same_pn_graph=True,\
            pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
            iso_check=True,name="Hard-Enzymes-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "proteins_hard":
        dataset = HardDataset(root, "proteins", size=500, positive=0.5, same_pn_graph=True,\
            pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.3, \
            iso_check=True,name="Hard-Proteins-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "mutag_hard":
        dataset = HardDataset(root, "mutag", size=500, positive=0.5, same_pn_graph=True,\
            pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
            iso_check=True,name="Hard-Mutag-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "aids_hard":
        dataset = HardDataset(root, "aids", size=500, positive=0.5, same_pn_graph=True,\
        pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.3, \
        iso_check=True,name="Hard-Aids-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "imdb_hard":
        dataset = HardDataset(root, "imdb-binary", size=500, positive=0.5, same_pn_graph=True,\
        pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-Imdb-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "cox2_hard":
        dataset = HardDataset(root, "cox2", size=500, positive=0.5, same_pn_graph=True,\
        pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-Cox2-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "firstmm_hard":
        dataset = HardDataset(root, "firstmm_db", size=500, positive=0.5, same_pn_graph=True,\
        pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-Firstmm-500", pre_transform=trans)
    elif dataset_name == "mutag_hard_i":
        dataset = HardInbalancedDataset(root, "mutag", size=500, positive=0.5, same_pn_graph=True,\
            pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
            iso_check=True,name="HardInbalanced-Mutag-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "enzymes_hard_i":
        dataset = HardInbalancedDataset(root, "enzymes", size=500, positive=0.5, same_pn_graph=True,\
            pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
            iso_check=True,name="HardInbalanced-Enzymes-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "proteins_hard_i":
        dataset = HardInbalancedDataset(root, "proteins", size=500, positive=0.5, same_pn_graph=True,\
            pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.3, \
            iso_check=True,name="HardInbalanced-Proteins-500", transform=trans, pre_transform=pre_trans)
    elif dataset_name == "ppi":
        dataset = RealDataset(root, "ppi",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="PPI-1000", pre_transform=None)
    elif dataset_name == "cifar10":
        dataset = RealDataset(root, "cifar10",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="CIFAR10-1000", pre_transform=None)
    elif dataset_name == "mnist":
        dataset = RealDataset(root, "mnist",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="MNIST-1000", pre_transform=None)
    elif dataset_name == "ogbmol":
        dataset = RealDataset(root, "ogbg-molhiv",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="OGBMOL-1000", pre_transform=trans)
    elif dataset_name == "ogbgpcba":
        dataset = RealDataset(root, "ogbg-molpcba",size = 1000, positive = 0.5, \
            pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
            neg_feat_type = "choice", iso_check = True , name="OGBPCBA-1000", pre_transform=trans)
    elif dataset_name == "ogbmol_hard":
        dataset = HardDataset(root, "ogbg-molhiv", size=500, positive=0.5, same_pn_graph=True,\
                pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
                iso_check=True,name="Hard-OGBMOL-500", pre_transform=trans)
    elif dataset_name == "ogbgpcba_hard":
        dataset = HardDataset(root, "ogbg-molpcba", size=500, positive=0.5, same_pn_graph=True,\
            pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
            iso_check=True,name="Hard-OGBPCBA-500", pre_transform=trans)
    elif dataset_name == "syn20":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=20,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=10, iso_check = True,
                feat_type="ones", feat_dim=10, name = "Syn20_10-500", transform=None, pre_transform=None)
    elif dataset_name == "syn30":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=30,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=15, iso_check = True,
                feat_type="ones", feat_dim=10, name = "Syn30_15-500", transform=None, pre_transform=None)
    elif dataset_name == "syn40":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=40,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=20, iso_check = True,
                feat_type="ones", feat_dim=10, name = "Syn40_20-500", transform=None, pre_transform=None)
    elif dataset_name == "syn50":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=50,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=25, iso_check = True,
                feat_type="ones", feat_dim=10, name = "Syn50_25-500", transform=None, pre_transform=None)
    elif dataset_name == "syn60":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=60,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=30, iso_check = False,
                feat_type="ones", feat_dim=10, name = "Syn60_30-500", transform=None, pre_transform=None)
    elif dataset_name == "syn70":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=70,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=35, iso_check = False,
                feat_type="ones", feat_dim=10, name = "Syn70_35-500", transform=None, pre_transform=None)
    elif dataset_name == "syn80":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=80,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=40, iso_check = False,
                feat_type="ones", feat_dim=10, name = "Syn80_40-500", transform=None, pre_transform=None)
    elif dataset_name == "syn90":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=90,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=45, iso_check = False,
                feat_type="ones", feat_dim=10, name = "Syn90_45-500", transform=None, pre_transform=None)
    elif dataset_name == "syn100":
        dataset = SynDataset(root, size=500, positive=250, graph_type=["ER","WS"], graph_sizes=100,\
                pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=50, iso_check = False,
                feat_type="ones", feat_dim=10, name = "Syn100_50-500", transform=None, pre_transform=None)

    if not n_fold is None:
        skf = StratifiedKFold(n_splits=n_fold)
        data_y = np.array([int(pair.y) for pair in dataset])
        if not random_seed is None:
            rng = default_rng(random_seed)
            permuted_id = rng.permutation(len(data_y))
            data_y = data_y[permuted_id]
        train_idx_list, test_index_list = [], []
        for train_index, test_index in skf.split(data_y, data_y):
            if not random_seed is None:
                train_index = permuted_id[train_index]
                test_index = permuted_id[test_index]
            train_idx_list.append(train_index)
            test_index_list.append(test_index)

        return dataset[train_idx_list[fold_idx]], dataset[test_index_list[fold_idx]]
    return dataset

if __name__ == "__main__":
    root = Path("D2Match/Dataset")
    if not root.exists():
        root.mkdir()
    feat_dim = 10
    #feat_type = "ones"
    from torch_geometric.transforms import Compose
    from assign_transform import ChordlessCycleTransfrom, DegreeCheckAssign
    trans = Compose([ChordlessCycleTransfrom(), DegreeCheckAssign()])
    '''
    dataset = SynDataset(root, size=1000, positive=500, graph_type=["ER","WS"], graph_sizes=40,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=15,
    feat_type="ones", feat_dim=feat_dim, name = "Syn40_15-1000", pre_transform=trans)

    dataset = RealDataset(root, "reddit-binary",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="Reddit-1000", pre_transform=trans)
    
    dataset = RealDataset(root, "enzymes",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="Enzymes-1000", pre_transform=trans)
    
    dataset = RealDataset(root, "proteins",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="Proteins-1000", pre_transform=trans)
    
    dataset = RealDataset(root, "reddit-binary",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="Reddit-1000", pre_transform=trans)
    
    dataset = RealDataset(root, "aids",size = 2000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 0.5, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="Aids-2000", pre_transform=trans)
    
    dataset = RealDataset(root, "imdb-binary",size = 2000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="Imdb-2000", pre_transform=trans)
    
    dataset = RealDataset(root, "mutag",size = 2000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 0.5, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="MUTAG-1000", pre_transform=trans)
    
    dataset = SynDataset(root, size=1000, positive=500, graph_type=["ER","WS"], graph_sizes=40,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=15, fix_edge_size=False,
    feat_type="ones", feat_dim=feat_dim, name = "Syn40_15-1000DensityNotFixed", pre_transform=trans)
    
    dataset = SynDataset(root, size=10, positive=5, graph_type=["ER","WS"], graph_sizes=40,\
    pos_subgraph_type="subtree", neg_subgraph_type="ER", subgraph_sizes=15,
    feat_type="ones", feat_dim=feat_dim, name = "Synss", pre_transform=trans)
    
    dataset = HardDataset(root, "enzymes", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
        iso_check=True,name="Hard-Enzymes-500", pre_transform=trans)
    
    dataset = HardDataset(root, "proteins", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.3, \
        iso_check=True,name="Hard-Proteins-500", pre_transform=trans)
    

    dataset = HardInbalancedDataset(root, "proteins", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.3, \
        iso_check=True,name="HardInbalanced-Proteins-500", pre_transform=trans)
    
    dataset = HardInbalancedDataset(root, "mutag", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
        iso_check=True,name="HardInbalanced-Mutag-500", pre_transform=trans)
    
    dataset = HardInbalancedDataset(root, "enzymes", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
        iso_check=True,name="HardInbalanced-Enzymes-500", pre_transform=trans)
    
    dataset = HardInbalancedDataset(root, "imdb-binary", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.5, \
        iso_check=True,name="HardInbalanced-Imdb-500", pre_transform=trans)
    
    dataset = HardDataset(root, "aids", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=0.3, \
        iso_check=True,name="Hard-Aids-500", pre_transform=trans)
    
    dataset = HardDataset(root, "imdb-binary", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-Imdb-500", pre_transform=trans)
    
    dataset = HardDataset(root, "cox2", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-Cox2-500", pre_transform=trans)
    
    dataset = HardDataset(root, "firstmm_db", size=500, positive=0.5, same_pn_graph=True,\
    pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-Firstmm-500", pre_transform=trans)
    
    dataset = RealDataset(root, "ppi",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="PPI-1000", pre_transform=None)
    
    dataset = RealDataset(root, "cifar10",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="CIFAR10-1000", pre_transform=None)
    
    dataset = RealDataset(root, "mnist",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="MNIST-1000", pre_transform=None)
    

    dataset = RealDataset(root, "ogbg-molhiv",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="OGBMOL-1000", pre_transform=trans)
    

    dataset = RealDataset(root, "ogbg-molpcba",size = 1000, positive = 0.5, \
        pos_subgraph_type = "subtree", neg_subgraph_type = "ER", subgraph_sizes = 15, subgraph_edge_sizes = None, \
        neg_feat_type = "choice", iso_check = True , name="OGBPCBA-1000", pre_transform=trans)
    
    dataset = HardDataset(root, "ogbg-molhiv", size=500, positive=0.5, same_pn_graph=True,\
        pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-OGBMOL-500", pre_transform=trans)
    
    dataset = HardDataset(root, "ogbg-molpcba", size=500, positive=0.5, same_pn_graph=True,\
        pos_subgraph_type="subtree", neg_subgraph_perturb=0.2, subgraph_sizes=15, \
        iso_check=True,name="Hard-OGBPCBA-500", pre_transform=trans)
    '''
    #enzymes, cox2, proteins, aids, reddit-binary, firstmm_db
    from torch_geometric.loader import DataLoader
    loader = DataLoader(dataset, batch_size = 4, follow_batch=['x_s', 'x_t'], shuffle = True)
    batch = next(iter(loader))
    print(batch)