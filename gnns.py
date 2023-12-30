import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn import MessagePassing

from gnn_convs import build_conv_model, EdgeConv, SingleConv

import numpy as np

class GeneralGNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.5, conv_type="GCN", skip="all"):
        super(GeneralGNN, self).__init__()
        self.dropout = dropout
        self.n_layers = num_layers
        self.skip = skip
        self.conv_type = conv_type
        
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))

        conv_model = build_conv_model(conv_type, 1)

        self.convs = nn.ModuleList()

        if skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                self.n_layers))

        for l in range(num_layers):
            if skip == 'all' or skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim

            self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (num_layers + 1)

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                    :i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)

                x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = self.post_mp(emb)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class EdgeGNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, edge_dim, dropout = 0.5):
        super(EdgeGNN, self).__init__()
        self.dropout = dropout
        self.n_layers = num_layers
        
        self.convs = nn.ModuleList()
        for l in range(num_layers-1):
            self.convs.append(EdgeConv(input_dim, hidden_dim, hidden_dim, edge_dim))
            input_dim = hidden_dim
        self.convs.append(EdgeConv(input_dim, hidden_dim, output_dim, edge_dim))
        
        
        
    def forward(self, data):
        x, edge_index, edge_feat = data.x, data.edge_index, data.edge_feat
        for i in range(len(self.convs)):
            x, edge_feat = self.convs[i](x, edge_index, edge_feat)
        
        return x, edge_feat

class DecoupledGNN(nn.Module):
    def __init__(self, conv_basis, num_layers, input_dim, hidden_dim, output_dim, conv_type = "sum", dropout = 0.5):
        super(DecoupledGNN, self).__init__()

        self.get_basis(conv_basis)
        self.conv_type = conv_type
        self.conv = SingleConv()

        self.trans = pyg_nn.MLP(num_layers=num_layers, in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim, dropout=dropout, batch_norm=False)
        self.basis_coef = torch.nn.Parameter(0.1*torch.randn(self.conv_basis.size(0)))

    def get_basis(self, conv_basis):
        max_degree = 0
        if isinstance(conv_basis, np.ndarray):
            conv_basis = torch.from_numpy(conv_basis)
            max_degree = conv_basis.shape[1]
        elif isinstance(conv_basis, list):
            if isinstance(conv_basis[0], np.ndarray):
                max_degree = max(list(map(lambda x:x.shape[0], conv_basis)))
                conv_basis = torch.stack(list(map(lambda x:F.pad(torch.from_numpy(x).float(),(max_degree - len(x), 0)), conv_basis)))
            elif isinstance(conv_basis[0], np.poly1d):
                conv_basis = list(map(lambda x:x.c, conv_basis))
                max_degree = max(list(map(lambda x:x.shape[0], conv_basis)))
                conv_basis = torch.stack(list(map(lambda x:F.pad(torch.from_numpy(x).float(),(max_degree - len(x), 0)), conv_basis)))
            elif isinstance(conv_basis[0], list):
                max_degree = max(list(map(lambda x:len(x), conv_basis)))
                conv_basis = torch.stack(list(map(lambda x:F.pad(torch.tensor(x, dtype=torch.float32),(max_degree - len(x), 0)), conv_basis)))
            else:
                raise RuntimeError("Convolution basis should be list or list of np.ndarray/np.poly1d")
        self.conv_basis = conv_basis
        self.max_degree = max_degree-1

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = [x]
        for i in range(self.max_degree):
            x = self.conv(x, edge_index)
            h.append(x)
        h = torch.flip(torch.stack(h), dims=[0])
        self.conv_basis = self.conv_basis.to(x.device)
        basis = self.conv_basis.repeat(h.size(1),1,1) @ h.permute(1,0,2)
        basis = basis.permute(1,0,2)
        basis = self.basis_coef.view(-1,1,1) * basis
        basis = self.trans(basis)

        return torch.cat(list(basis), dim=-1)


            
        
        