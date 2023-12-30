import numpy as np
from torch_cluster import neighbor_sampler
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt

def Nomial(order):
    x = np.poly1d([1,0])
    return [x**i for i in range(order)]

def rooted_tree(data: Data, node, depth=3):
    num_nodes = data.x.size(0)
    if not isinstance(node,torch.Tensor):
        node = torch.tensor([node])

    tree_edge = []
    node_index = 0
    leaf_index = 1

    for i in range(depth):
        neighbors = []
        for n in node:
            mask = pyg_utils.index_to_mask(n, num_nodes)
            neighbor = data.edge_index[1][mask[data.edge_index[0]]]
            tree_edge.append(torch.tensor([[node_index]*len(neighbor),list(range(leaf_index, leaf_index+len(neighbor)))]))
            leaf_index += len(neighbor)
            node_index += 1
            neighbors.append(neighbor)
        
        node = torch.cat(neighbors)

    x = torch.ones(node_index,1)
    edge_index_1 = torch.cat(tree_edge, dim=1)
    edge_index_2 = edge_index_1[[1,0]]
    edge_index = torch.cat([edge_index_1,edge_index_2],dim=1)
    return pyg_utils.to_networkx(Data(x=x,edge_index=edge_index))

from torch_geometric.data import Data
def from_data_pair(data_pair):
    data_t = Data(data_pair.x_t, data_pair.edge_index_t)
    data_s = Data(data_pair.x_s, data_pair.edge_index_s)
    return data_t, data_s

def to_graph_pair(data_pair):
    data_t, data_s = from_data_pair(data_pair)
    graph_t, graph_s = pyg_utils.to_networkx(data_t, node_attrs=["x"]), pyg_utils.to_networkx(data_s, node_attrs=["x"])
    return nx.Graph(graph_t), nx.Graph(graph_s)

from networkx.algorithms import isomorphism
def is_subgraph_iso(graph_t, graph_s):
    GM = isomorphism.GraphMatcher(graph_t, graph_s)
    return GM.subgraph_is_isomorphic()

import matplotlib.pyplot as plt 
def plot_graph(graph, name="plot.png"):
    plt.figure()
    nx.draw(graph)
    plt.savefig(name)

import seaborn as sns
def plot_mat(mat, name="mat.png"):
    plt.figure()
    sns.heatmap(mat)
    plt.savefig(name)

from torch_geometric.nn import global_max_pool
def drop_adj(edge_index, p):
    row, col = edge_index
    prob = torch.rand(col.size(0), device=edge_index.device)

    max_node_p = global_max_pool(prob, col)

    revised_node_p = torch.relu(p-max_node_p+1e-10)
    mask = (prob + revised_node_p[col])>=p
    
    col = col[mask]
    row = row[mask]
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


import threading
import time
class MyThread(threading.Thread):
    def __init__(self, target, args=()):
        """
            因为threading类没有返回值,因此在此处重新定义MyThread类,使线程拥有返回值
        """
        super(MyThread, self).__init__()
        self.func = target
        self.args = args

    def run(self):
        # 接受返回值
        self.result = self.func(*self.args)

    def get_result(self):
        # 线程不结束,返回值为None
        try:
            return self.result
        except Exception:
            return None


def limit_time(func, timeout, granularity, *args):
    """
        timeout 最大允许执行时长, 单位:秒
        granularity 轮询间隔，间隔越短结果越精确同时cpu负载越高
        return 未超时返回被装饰函数返回值,超时则返回 False
    """
    thre_func = MyThread(target=func, args=args)
    thre_func.setDaemon(True)
    thre_func.start()
    sleep_num = int(timeout // granularity)
    for i in range(0, sleep_num):
        infor = thre_func.get_result()
        if infor is not None:
            return infor
        else:
            time.sleep(granularity)
    return None


'''
edge_index = torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,0,2,0,1,0]])
x = torch.ones(4,1)
data = Data(x,edge_index)
g = rooted_tree(data, 0)
plt.figure()
nx.draw_planar(g)
plt.savefig("~/NeuralMatch/plot.png")
'''



def feat_assign(graph_t, graph_s):
    x_t = np.array(list(map(lambda x: x[1],list(graph_t.nodes(data="x")))))
    x_s = np.array(list(map(lambda x: x[1],list(graph_s.nodes(data="x")))))

    assign = x_t @ x_s.T
    return assign

def cycle_assign(graph_t, graph_s):
    x_t = np.array(list(map(lambda x: x[1],list(graph_t.nodes(data="node_type")))))
    x_s = np.array(list(map(lambda x: x[1],list(graph_s.nodes(data="node_type")))))

    assign = x_t.reshape([-1,1]) == x_s.T.reshape([1,-1])
    return assign


def neighbor_graph_assign(graph_t, graph_s):
    t_size = graph_t.number_of_nodes()
    s_size = graph_s.number_of_nodes()
    if hasattr(graph_t, "true_nodes"):
        graph_t = graph_t.subgraph(range(graph_t.true_nodes))
        graph_s = graph_s.subgraph(range(graph_s.true_nodes))

    t_neighbor_graph = []
    for i in list(graph_t.nodes):
        neigh = list(graph_t.neighbors(i))
        neigh.append(i)
        t_neighbor_graph.append(nx.Graph(graph_t.subgraph(neigh)))
    
    s_neighbor_graph = []
    for i in list(graph_s.nodes):
        neigh = list(graph_s.neighbors(i))
        #neigh.append(i)
        s_neighbor_graph.append(nx.Graph(graph_s.subgraph(neigh)))

    assign = np.zeros([graph_t.number_of_nodes(), graph_s.number_of_nodes()])
    for i in range(len(t_neighbor_graph)):
        for j in range(len(s_neighbor_graph)):
            assign[i,j] = is_subgraph_iso(t_neighbor_graph[i], s_neighbor_graph[j])

    bed = np.ones([t_size, s_size])
    bed[0:graph_t.number_of_nodes(),:][:,0:graph_s.number_of_nodes()] = assign
    return bed


def check_assign(assign_mat, test_time=0):
    result = np.sum(np.max(assign_mat, axis=1))>=assign_mat.shape[1]
    result = result * (np.sum(np.max(assign_mat, axis=0))==assign_mat.shape[1])

    for i in range(test_time):
        size = np.random.randint(1, assign_mat.shape[1]-1, [1])
        idx = np.random.choice(np.arange(0, assign_mat.shape[1]), size, replace=False)
        sampled_assign_mat = assign_mat[:,idx]

        result = result * np.sum(np.max(sampled_assign_mat, axis=1))>=size
    return result



def iso_check(graph_t, graph_s, iter=3, feat_check=True, cycle_check=True, neighbor_graph=False):
    num_t, num_s = graph_t.number_of_nodes(), graph_s.number_of_nodes()
    assign = np.ones([num_t, num_s])
    if feat_check:
        assign = assign * feat_assign(graph_t, graph_s)
    if neighbor_graph:
        assign = assign * neighbor_graph_assign(graph_t, graph_s)
    if cycle_check:
        assign = assign * cycle_assign(graph_t, graph_s)
    mat = None

    for i in range(iter):
        assign = dual_check(graph_t, graph_s, assign)
        #assign = cycle_check(graph_t, graph_s, assign, root=i%num_s)
    
    result = np.linalg.matrix_rank(np.random.rand(*assign.shape) * assign) == assign.shape[1]
    return result


def degree_mat(graph_t, graph_s):
    num_t, num_s = graph_t.number_of_nodes(), graph_s.number_of_nodes()
    assign = np.ones([num_t, num_s])
    mat = {}

    neigh_t = []
    neigh_s = []
    for i in range(num_t):
        neigh_t.append(list(graph_t.neighbors(i)))
    for j in range(num_s):
        neigh_s.append(list(graph_s.neighbors(j)))

    A_t = nx.to_numpy_array(graph_t)
    A_s = nx.to_numpy_array(graph_s)
    for i in range(num_t):
        for j in range(num_s):
            if len(neigh_t[i]) == 0:
                assign[i,j] = 0
            elif len(neigh_s[j]) == 0:
                a = 1
            else:
                deg_t = A_t[neigh_t[i], :][:, neigh_t[i]].sum(axis=1)
                deg_s = A_s[neigh_s[j], :][:, neigh_s[j]].sum(axis=1)
                mat[(i,j)] = assign[neigh_t[i], :][:, neigh_s[j]] * (deg_t.reshape([-1,1])>=deg_s.reshape([1,-1])).astype(float)
                assign[i,j] = min(np.sum(mat[(i,j)], axis=0))>0
    return assign, mat

def graph_pair(data_pair):
    data_t, data_s = from_data_pair(data_pair)
    graph_t, graph_s = pyg_utils.to_networkx(data_t, node_attrs=["x"]), pyg_utils.to_networkx(data_s, node_attrs=["x"])
    return nx.Graph(graph_t), nx.Graph(graph_s)


def dual_check(graph_t, graph_s, assign, sample = 2):
    num_t, num_s = graph_t.number_of_nodes(), graph_s.number_of_nodes()

    neigh_t = []
    neigh_s = []
    for i in range(num_t):
        neigh_t.append(list(graph_t.neighbors(i)))
    for j in range(num_s):
        neigh_s.append(list(graph_s.neighbors(j)))

    result = np.ones([num_t, num_s])

    for i in range(num_t):
        for j in range(num_s):
            if assign[i,j] > 0:
                mat = assign[neigh_t[i],:][:, neigh_s[j]]
                if len(neigh_s[j]) == 0:
                    result[i,j] = 0
                elif len(neigh_t[i]) == 0:
                    result[i,j] = 0
                else:
                    check = min(np.sum(mat, axis=0))>0
                    check *= sum(np.max(mat, axis=1))>=mat.shape[1]

                    for k in range(sample):
                        neigh_s_ = np.random.choice(neigh_s[j], np.random.randint(max(1,len(neigh_s[j])-1))+1, replace=False)
                        mat = assign[neigh_t[i],:][:, neigh_s_]
                        check_ = np.sum(np.max(mat, axis=1))>=len(neigh_s_)
                        if check==True and check_ == False:
                            #print(i,j)
                            #print(neigh_s_)
                            check = False
                            break
                    #check = np.linalg.matrix_rank(np.random.rand(*mat.shape) * mat) == mat.shape[1]
                    if check == False:
                        #print(i,j)
                        result[i,j] = 0
    return assign * result
        


def cycle_check(graph_t, graph_s, assign, root=0):
    graph_t, graph_s = nx.Graph(graph_t), nx.Graph(graph_s)
    cycle_basis = nx.cycle_basis(graph_s, root)

    A_t = nx.to_numpy_array(graph_t)

    for cycle in cycle_basis:
        j = cycle[0]
        for i in range(assign.shape[0]):
            if assign[i, j]>0:
                id = np.zeros(assign.shape[0])
                id[i] = 1
                
                for l in range(1, len(cycle)):
                    id = A_t @ id
                    id = id * assign[:, cycle[l]]
                    id[i] = 0
                id = A_t @ id
                if id[i] == 0:
                    #print(i,j)
                    assign[i, j] = 0
    return assign

from chordless import ChordlessCycles
def add_cycle_nodes(graph, length = 4):
    if not hasattr(graph, "true_nodes"):
        graph.true_nodes = graph.number_of_nodes()
    
    subgraph = graph.subgraph(range(graph.true_nodes))
    chordless_cycles = ChordlessCycles(subgraph, length+1)
    cycles = []
    for c in chordless_cycles:
        if len(c) == length:
            cycles.append(c)
    
    node_id = max(graph.nodes)+1
    for c in cycles:
        graph.add_node(node_id, x = [1.0] * len(graph.nodes[0]["x"]), node_type = length)
        for n in c:
            graph.add_edge(node_id, n)
        node_id+=1
    return graph