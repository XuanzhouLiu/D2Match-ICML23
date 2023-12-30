import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations


def DegreeLabeling(g):#white 0 , black 1
  for u in g.nodes():
    g.nodes[u]['color'] = 0
    g.nodes[u]['degree'] = g.degree(u)
  
  v = -1
  for i in range(len(g)):
    min_degree = len(g)
    for x in g.nodes():
      if g.nodes[x]['color'] == 0 and g.nodes[x]['degree'] < min_degree:
        v = x
        min_degree = g.degree(x)
    
    g.nodes[v]['label'] = i
    g.nodes[v]['color'] = 1

    for u in g[v]:
      if g.nodes[u]['color'] == 0:
        g.nodes[u]['degree'] -= 1

  return g

def Triplets(g):
  T,C = list() , list()

  for u in g.nodes():
    for x,y in list(permutations(g[u], 2)):
      if g.nodes[u]['label'] < g.nodes[x]['label'] < g.nodes[y]['label']:
        if y in g[x]:
          C.append([x ,u , y])
        else:
          T.append([x , u , y])
  return T,C


def BlockNeighbors (u, g):
  for v in g[u]:
    g.nodes[v]['blocked'] += 1

def UnblockNeighbors (u, g):
  for v in g[u]:
    if g.nodes[v]['blocked'] > 0:
      g.nodes[v]['blocked'] -= 1

def CC_Visit (p,C,key,g, length):
  u_t = p[-1]
  BlockNeighbors(u_t ,g)

  for v in g[u_t]:
    if g.nodes[v]['label'] > key and g.nodes[v]['blocked'] == 1 and len(p)<length:
      p_ = p + [v]
      if p[0] in g[v]:
        C.append(p_)
      else:
        CC_Visit(p_ , C , key , g, length)
  
  UnblockNeighbors(u_t , g)

  return C

def ChordlessCycles(g, length = 4):
  g = DegreeLabeling(g)
  T,C  = Triplets(g)

  for u in g.nodes():
    g.nodes[u]['blocked'] = 0
  
  while len(T) != 0:
    p = T[0]# p = (x,u,y)
    del T[0]
    u = p[1]
    BlockNeighbors(u,g)
    key = g.nodes[u]['label']
    C = CC_Visit(p , C , key , g, length)
    UnblockNeighbors(u,g)

  for (n,d) in g.nodes(data=True):
    if "label" in d:
      del d["label"]
    if "blocked" in d:
      del d['blocked']
    if "degree" in d:
      del d['degree']
    if "color" in d:
      del d['color']
  return C
'''
g = nx.generators.gnp_random_graph(40,0.3)
nx.draw_networkx(g)
plt.show()

Cs = ChordlessCycles(graph_s)
Ct = ChordlessCycles(graph_t)
print(len(Cs))

Cs5 = []
for c in Cs:
    if len(c)==5:
        Cs5.append(c)

Cs0 = []
for c in Cs:
    if 0 in Cs and 3 in Cs and 4 in Cs:
        Cs0.append(c)

Ct5 = []
for c in Ct:
    if len(c)==5:
        Ct5.append(c)

assign[np.unique(Ct5),:][:,list(set(range(assign.shape[1])).difference(np.unique(Cs)))]=0
'''