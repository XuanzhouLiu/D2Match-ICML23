import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import  pad_sequence 
from torch_geometric.data import Data, Batch
import torch_geometric.utils as pyg_utils

import pytorch_lightning as pl
import torchmetrics

from gnn_convs import SingleConv, build_conv_model
from gnns import GeneralGNN
from utils import *
import time

class TestE2E(pl.LightningModule):
    def __init__(self, num_layers, input_dim, hidden_dim, dropout=0.5, lr=0.01, gnn_update=True, shared_gnn=False, subtree_update=True, learnable_subtree=True, gnn_interact=True, init_trans=True, sample_num=2, loss="mse", aggr="mean", **kwargs):
        """
        """
        super(TestE2E, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        
        self.loss = loss
        self.aggr = aggr
        if self.aggr == "learnable":
            out_dim = 0
            if gnn_update:
                out_dim+=1
            if subtree_update:
                out_dim+=1
            self.clf_model = nn.Linear(out_dim, 1)

        # model architecture settings
        self.init_transform = init_trans

        # gnn matching module
        self.gnn_update = gnn_update
        self.gnn_interact = gnn_interact if (gnn_update and subtree_update) else False

        # subtree matching module
        self.subtree_update = subtree_update
        self.learnable_subtree = learnable_subtree if subtree_update else False
        self.sample_num = sample_num
        
        # neural settings
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.init_transform:
            self.init_trans = pyg_nn.MLP([input_dim, hidden_dim, hidden_dim])

        if self.subtree_update:
            self.rw_max_conv = SingleConv(conv_type="random", aggr="max")
            self.sum_conv = SingleConv(aggr="add")
            self.max_conv = SingleConv(aggr="max")
            self.mean_conv = SingleConv(aggr="mean")
            
            self.sigmoid_t = nn.ModuleList()
            self.sigmoid_s = nn.ModuleList()
            for i in range(self.num_layers):
                self.sigmoid_t.append(LearnableSigmoid(0.99,1000))
                self.sigmoid_s.append(LearnableSigmoid(0.99,1000))

            self.out_sigmoid_1 = LearnableSigmoid(0.99,1000)
            self.out_sigmoid_2 = LearnableSigmoid(0.99,1000)

        if self.gnn_interact:
            self.inter_feat_trans = nn.ModuleList()
            for i in range(self.num_layers):
                if i==0:
                    self.inter_feat_dim = 2*input_dim if not self.init_transform else 2*hidden_dim
                else:
                    self.inter_feat_dim = 3*hidden_dim
                self.inter_feat_trans.append(pyg_nn.MLP([self.inter_feat_dim, 2*hidden_dim, hidden_dim], dropout=dropout))

        if self.gnn_update:
            
            if self.gnn_interact:
                self.inter_feat_trans = nn.ModuleList()
                for i in range(self.num_layers):
                    if i==0:
                        self.inter_feat_dim = 2*input_dim if not self.init_transform else 2*hidden_dim
                    else:
                        self.inter_feat_dim = 3*hidden_dim
                    self.inter_feat_trans.append(pyg_nn.MLP([self.inter_feat_dim, 2*hidden_dim, hidden_dim], dropout=dropout))

            gnn_conv = build_conv_model("GIN", 1)
            self.sub_convs = nn.ModuleList()
            for i in range(self.num_layers):
                if i==0 and not self.init_transform:
                    gnn_input_dim = input_dim + hidden_dim if self.gnn_interact else input_dim
                else:
                    gnn_input_dim = 2 * hidden_dim if self.gnn_interact else hidden_dim

                self.sub_convs.append(gnn_conv(gnn_input_dim, hidden_dim))

            self.attention_layer_s = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.attention_layer_s.weight)

            if shared_gnn:
                self.target_convs = self.sub_convs
                self.attention_layer_t = self.attention_layer_s
            else:
                self.target_convs = nn.ModuleList()
                for i in range(self.num_layers):
                    if i==0 and not self.init_transform:
                        gnn_input_dim = input_dim + hidden_dim if self.gnn_interact else input_dim
                    else:
                        gnn_input_dim = 2 * hidden_dim if self.gnn_interact else hidden_dim

                    self.target_convs.append(gnn_conv(gnn_input_dim, hidden_dim))

                self.attention_layer_t = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                torch.nn.init.xavier_uniform_(self.attention_layer_t.weight)


            self.attention_layer_st = torch.nn.Bilinear(self.hidden_dim, self.hidden_dim, self.hidden_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.attention_layer_st.weight)
            self.cat_layer_st = torch.nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.cat_layer_st.weight)
            self.bias = torch.nn.Parameter(torch.Tensor(self.hidden_dim, 1))
            torch.nn.init.xavier_uniform_(self.bias)

            self.out_trans = pyg_nn.MLP([hidden_dim, 2*hidden_dim, 1], batch_norm=False)#No batch norm

        self.lr = lr
        self.train_acc = torchmetrics.Accuracy(num_classes=2)
        self.val_acc = torchmetrics.Accuracy(num_classes=2)
        self.total_loss = torchmetrics.SumMetric()
        self.total_mse_loss = torchmetrics.SumMetric()

        self.tmp = torch.nn.Parameter(torch.ones(1))


    def forward(self, data_pair):
        data_t = Data(data_pair.x_t, data_pair.edge_index_t)
        data_s = Data(data_pair.x_s, data_pair.edge_index_s)

        # initialize node feature
        if self.init_transform:
            x_t = self.init_trans(data_t.x.float())
            x_s = self.init_trans(data_s.x.float())
        else:
            x_t = data_t.x.float()
            x_s = data_s.x.float()

        # change batched features to list of features,  x_*_list: batch * *_max_nodes * hidden_dim
        x_t_list, x_t_mask = pyg_utils.to_dense_batch(x_t, data_pair.x_t_batch)
        x_s_list, x_s_mask = pyg_utils.to_dense_batch(x_s, data_pair.x_s_batch)
        
        # initialize assign matrix
        assign_mat = self.init_assign(x_t_list, x_s_list)

        assign_mat = assign_mat * self.degree_check_assign(data_pair, assign_mat.size())
        #if not self.subtree_update:
        #    assign_mat = torch.ones(assign_mat.size()).to(assign_mat.device)

        inter_feature = None

        for i in range(self.num_layers):
            if self.subtree_update:
                assign_mat = self.update_subtree_assign(i, assign_mat, x_t, x_t_list, x_t_mask, x_s, x_s_list, x_s_mask, data_pair.edge_index_t, data_pair.edge_index_s, data_pair.x_t_batch, data_pair.x_s_batch)
            if self.gnn_update:
                x_t, x_t_list, x_s, x_s_list, inter_feature = self.update_x(i, assign_mat, inter_feature, x_t, x_t_list, x_t_mask, x_s, x_s_list, x_s_mask, data_pair.edge_index_t, data_pair.edge_index_s, data_pair.x_t_batch, data_pair.x_s_batch)

            
        #if self.gnn_interact:
        #    assign_mat = assign_mat * inter_feature.mean(dim=-1)

        # eliminate fake nodes
        assign_mat = x_t_mask.unsqueeze(2) * assign_mat * x_s_mask.unsqueeze(1)

        score = []
        count = 0
        # similarity score of assign matrix
        if self.subtree_update:
            if self.learnable_subtree:
                result_1 = torch.sum(torch.max(assign_mat, dim=1)[0], dim=1)/data_pair.s_nodes.view(-1)
                result_2 = torch.sum(torch.max(assign_mat, dim=1)[0], dim=1)/data_pair.s_nodes.view(-1)
                result = self.out_sigmoid_1(result_1) * self.out_sigmoid_2(result_2)
            else:
                result_1 = torch.sum(torch.max(assign_mat, dim=1)[0], dim=1)/data_pair.s_nodes.view(-1)>=1
                result_2 = torch.sum(torch.max(assign_mat, dim=1)[0], dim=1)/data_pair.s_nodes.view(-1)==1
                result = (result_1 * result_2).float()
            
            score.append(result.unsqueeze(1))

        if self.gnn_update:
            context_s = torch.tanh(pyg_nn.global_mean_pool(self.attention_layer_s(x_s), data_pair.x_s_batch))
            context_t = torch.tanh(pyg_nn.global_mean_pool(self.attention_layer_t(x_t), data_pair.x_t_batch))

            sigmoid_scores_t = torch.sigmoid(x_t_list @ context_t.unsqueeze(2))
            e_t = (x_t_list.permute(0,2,1)@sigmoid_scores_t).squeeze()
            sigmoid_scores_s = torch.sigmoid(x_s_list @ context_s.unsqueeze(2))
            e_s = (x_s_list.permute(0,2,1)@sigmoid_scores_s).squeeze()

            gnn_score = F.relu(self.attention_layer_st(e_t,e_s)+self.cat_layer_st(torch.cat([e_t,e_s], dim=-1))+self.bias.squeeze())
            gnn_score = self.out_trans(gnn_score)

            inter_score = 0
            if self.gnn_interact:
                inter_feature = assign_mat * inter_feature.sum(dim=-1)
                inter_score = inter_feature.view(inter_feature.size(0),-1).mean(dim=-1).view(-1,1)
            gnn_score = torch.sigmoid(gnn_score + inter_score)
            score.append(gnn_score)

        if self.aggr == "mul":
            out = torch.prod(torch.cat(score, dim=-1), dim=-1)
            return out
        elif self.aggr == "sep":
            out = score
            return out

    def criterion(self, out, y):
        if self.aggr == "sep":
            loss = 0
            if self.subtree_update:
                loss = F.l1_loss(out[0].squeeze(), y.float()) + loss
            if self.gnn_update:
                if self.loss == "mse":
                    loss = F.mse_loss(out[-1].squeeze(), y.float()) + loss
                    self.total_mse_loss.update(loss)
                elif self.loss == "nll":
                    loss = F.binary_cross_entropy(out[-1].pop(0).squeeze(), y.float()) + loss
                elif self.loss == "mae":
                    loss = F.l1_loss(out[-1].pop(0).squeeze(), y.float()) + loss
            return loss

        if self.loss == "mse":
            loss = F.mse_loss(out, y.float())
            self.total_mse_loss.update(loss)
        elif self.loss == "mae":
            loss = F.l1_loss(out, y.float())
        elif self.loss == "nll":
            loss = F.binary_cross_entropy(out, y.float())
        return loss

    def predict(self, out):
        if self.aggr == "sep":
            out = torch.cat(out, dim=-1)
            predict = torch.prod((out>=0.5).float(), dim=-1).squeeze()
            return predict.long()

        predict = (out>=0.5).squeeze()
        return predict

    def training_step(self, batch, batch_idx):
        #t = time.time()
        opt = self.optimizers()
       
        out = self(batch)
        #print(out)
        opt.zero_grad()
        loss = self.criterion(out, batch.y)
        self.total_loss.update(loss)
        if loss.requires_grad:
            self.manual_backward(loss)
            opt.step()
        self.log("train_loss", loss, prog_bar=True)

        predict = self.predict(out)
        #print("total",time.time()-t)

        self.log("train acc", self.train_acc(predict, batch.y))
        #print(self.target_gnn.basis_coef.data)
        return
    
    def training_epoch_end(self, out):
        self.log("total train acc", self.train_acc.compute(), prog_bar=True)
        if self.loss == "mse":
            self.log("total train loss", self.total_mse_loss.compute(), prog_bar=True)
        else:
            self.log("total train loss", self.total_loss.compute(), prog_bar=True)
        self.train_acc.reset()
        self.total_loss.reset()
        self.total_mse_loss.reset()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        predict = self.predict(out)
        self.log("val acc", self.val_acc((predict>0.5).squeeze(), batch.y))
    
    def validation_epoch_end(self, out):
        self.log("total val acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.aggr == "learnable":
            clf_opt = torch.optim.Adam(self.clf_model.parameters(), lr=self.lr)
            return opt, clf_opt
        return opt

    def init_assign(self, x_t_list, x_s_list):
        #normalize feature to calculate cosine similarity
        normed_x_t_list = (x_t_list +1e-12) / (torch.norm(x_t_list, dim=-1)+1e-12).unsqueeze(-1)
        normed_x_s_list = (x_s_list +1e-12) / (torch.norm(x_s_list, dim=-1)+1e-12).unsqueeze(-1)

        assign_mat = normed_x_t_list @ normed_x_s_list.permute(0,2,1)
        if not self.learnable_subtree:
            assign_mat = assign_mat.bool().float()

        return F.relu(assign_mat)

    def degree_check_assign(self, data_pair, size):
        if hasattr(data_pair, "degree_assign"):
            #t = time.time()
            degree_assign_mat = torch.zeros(size, device = data_pair.x_t.device)
            for i, assign in enumerate(data_pair.degree_assign):
                degree_assign_mat[i,0:assign.shape[0],0:assign.shape[1]] = torch.from_numpy(assign)
            #print("assign:",time.time()-t)
            return degree_assign_mat
        return torch.ones(size, device = data_pair.x_t.device)

    def update_inter_feature(self, i, x_t_list, x_s_list, inter_feature):
        # use gnn feature to update interact feature
        if i == 0:
            inter_feature_size = [x_t_list.size(0), x_t_list.size(1), x_s_list.size(1), self.hidden_dim]
        else:
            inter_feature_size = inter_feature.size()

        x_t_list_ = x_t_list.unsqueeze(2).expand(x_t_list.size(0), x_t_list.size(1), x_s_list.size(1), x_t_list.size(2))
        x_s_list_ = x_s_list.unsqueeze(1).expand(x_t_list.size(0), x_t_list.size(1), x_s_list.size(1), x_t_list.size(2))
        
        if i == 0:
            inter_feature = torch.cat([x_t_list_, x_s_list_], dim=3)
        else:
            inter_feature = torch.cat([inter_feature, x_t_list_, x_s_list_], dim=3)

        inter_feature = self.inter_feat_trans[i](inter_feature.view(-1,inter_feature.size(-1)))
        inter_feature = inter_feature.view(*inter_feature_size)
        #assign_mat = (assign_mat * inter_feature.sum(dim=-1))
        return inter_feature

    def update_subtree_assign(self, i, assign_mat, x_t, x_t_list, x_t_mask, x_s, x_s_list, x_s_mask, edge_index_t, edge_index_s, x_t_batch, x_s_batch):
        # view the assign matrix as node feature, convolute respectively
        t_inter_feat = assign_mat[x_t_mask].view(x_t.size(0), x_s_mask.size(1))
        s_inter_feat = assign_mat.permute(0,2,1)[x_s_mask].view(x_s.size(0), x_t_mask.size(1))
        #t_inter_feat: t_nodes x s_max_nodes

        for sample in range(self.sample_num):
            if sample == 0:
                sampled_edge_index_s = edge_index_s
            else:
                sampled_edge_index_s = drop_adj(edge_index_s, 0.5)
            s_feat_1 = self.rw_max_conv(s_inter_feat, sampled_edge_index_s)
            # s_feat_1: s_nodes x t_max_nodes
            t_feat_1 = pyg_utils.to_dense_batch(s_feat_1, x_s_batch)[0]
            # t_feat_1: batch x s_max_nodes x t_max_nodes
            t_feat_1 = t_feat_1.permute(0,2,1)[x_t_mask].view(-1, x_s_mask.size(1))
            # t_feat_1: t_nodes x s_max_nodes
            t_feat_1 = self.sum_conv(t_feat_1, edge_index_t).view(-1, x_s_mask.size(1))
            # t_feat_1: t_nodes x s_max_nodes

            if sample == 0:
                if self.learnable_subtree:
                    t_feat = self.sigmoid_t[i](t_feat_1)
                else:
                    t_feat = (t_feat_1>=1-1e-6).float()
            else:
                if self.learnable_subtree:
                    t_feat = t_feat * self.sigmoid_t[i](t_feat_1)
                else:
                    t_feat = t_feat * (t_feat_1>=1-1e-6).float()

        t_feat_2 = self.max_conv(t_inter_feat, edge_index_t)
        # t_feat_2: t_nodes x s_max_nodes
        s_feat_2 = pyg_utils.to_dense_batch(t_feat_2, x_t_batch)[0]
        # s_feat_2: batch x t_max_nodes x s_max_nodes
        s_feat_2 = s_feat_2.permute(0,2,1)[x_s_mask].view(-1, x_t_mask.size(1))
        # s_feat_2: s_nodes x t_max_nodes
        s_feat_2 = self.mean_conv(s_feat_2, edge_index_s).view(-1, x_t_mask.size(1))
        # s_feat_2: s_nodes x t_max_nodes
        if self.learnable_subtree:
            s_feat_2 = self.sigmoid_s[i](s_feat_2)
        else:
            s_feat_2 = (s_feat_2>=1-1e-6).float()

        assign_1 = pyg_utils.to_dense_batch(t_feat, x_t_batch)[0]
        # batch * t_max_nodes x s_max_nodes
        assign_2 = pyg_utils.to_dense_batch(s_feat_2, x_s_batch)[0]
        # batch x s_max_nodes x t_max_nodes
        assign_mat = assign_1 * assign_2.permute(0,2,1) * assign_mat

        return assign_mat

    def update_x(self, i, assign_mat, inter_feature, x_t, x_t_list, x_t_mask, x_s, x_s_list, x_s_mask, edge_index_t, edge_index_s, x_t_batch, x_s_batch):
        if self.gnn_interact:
            inter_feature = self.update_inter_feature(i, x_t_list, x_s_list, inter_feature)
        
            inter_feature_ = inter_feature * assign_mat.unsqueeze(-1)
            t_assign_feat = pyg_nn.global_mean_pool(inter_feature_.permute(0,2,1,3)[x_s_mask], x_s_batch)[x_t_mask]
            s_assign_feat = pyg_nn.global_mean_pool(inter_feature_[x_t_mask], x_t_batch)[x_s_mask]
        # *_nodes x hidden_dim
            t_feat = torch.cat([x_t,t_assign_feat], dim=-1)
            s_feat = torch.cat([x_s,s_assign_feat], dim=-1)
        else:
            t_feat = x_t
            s_feat = x_s

        x_t = self.target_convs[i](t_feat, edge_index_t)
        x_s = self.sub_convs[i](s_feat, edge_index_s)

        x_t_list = pyg_utils.to_dense_batch(x_t, x_t_batch)[0]
        x_s_list = pyg_utils.to_dense_batch(x_s, x_s_batch)[0]
        return x_t, x_t_list, x_s, x_s_list, inter_feature

class LearnableSigmoid(nn.Module):
    def __init__(self, init_value, scale = 5):
        super().__init__()
        self.init_value = torch.tensor(scale*init_value).float()
        self.offset = nn.Parameter(self.init_value)
        self.mlp = pyg_nn.MLP([1,4,1])
        self.scale = torch.tensor(scale).float()
        self.sc = nn.Parameter(self.scale)

    def forward(self, x):
        #size = x.size()
        #x = self.mlp(x.view(-1,1))
        #torch.sigmoid(x.view(size)*self.scale)
        return torch.sigmoid(x*self.sc-self.offset)

    #plot_mat(assign_mat[0][x_t_mask[0]][:,x_s_mask[0]].cpu().numpy())