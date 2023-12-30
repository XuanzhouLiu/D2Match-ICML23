# -*- coding: utf-8 -*-
"""
@author: 34753
"""
import argparse
from pickle import TRUE
from e2e_model import TestE2E
from dataset import SynDataset, RealDataset, PairData, get_dataset
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

from pathlib import Path

from assign_transform import DegreeCheckAssign, ChordlessCycleTransfrom, NoFeatureTransfrom
from config import parse_encoder



def main():
    parser = parse_encoder()
    args = parser.parse_args()
    root = Path(args.root)

    gpu_idx = args.gpu_idx
    random_seed = args.split_seed
    if random_seed == -1:
        random_seed = None
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_samples = args.num_samples

    model_name = args.model_name
    dataset_name = args.dataset_name

    fold_num = args.fold_num
    fold_idx = args.fold_idx

    max_epochs = args.max_epoch
    batch_size = args.batch_size

    exp_name = args.exp_name
    if not exp_name == "":
        exp_name = "{}_{}_{}".format(exp_name, model_name, dataset_name)
    else:
        exp_name = "{}_{}".format(model_name, dataset_name)
    
    if args.version != -1:
        version = args.version
        if (root/ "Logs" / exp_name / "version_{}".format(version) / "fold_{}".format(fold_idx)).exists():
            print("Experiment already exists!")
            return
    else:
        version = 0
        while (root/ "Official_Logs" / exp_name / "version_{}".format(version) / "fold_{}".format(fold_idx)).exists():
            version += 1

    pre_trans = Compose([ChordlessCycleTransfrom(), DegreeCheckAssign()]) if args.cc else None
    trans= NoFeatureTransfrom() if args.no_feature else None

    if not root.exists():
        root.mkdir()
    if not (root/"Dataset").exists():
        (root/"Dataset").mkdir()

    train_set, val_set = get_dataset(root/"Dataset", dataset_name, trans, pre_trans, fold_num, fold_idx, random_seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, follow_batch=['x_s', 'x_t'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, follow_batch=['x_s', 'x_t'], shuffle=False)

    feat_dim = train_set[0].x_t.size(1)
    model = TestE2E(num_layers, feat_dim, hidden_dim, dropout=0.5, lr=3e-4, gnn_update=args.no_gnn_update, gnn_interact=args.no_gnn_interact, shared_gnn=False, subtree_update=args.no_subtree_update, learnable_subtree=True, init_trans=False,sample_num=num_samples, loss="mse", aggr="sep")

    logger = TensorBoardLogger(root / "Logs", name=exp_name, version=version, sub_dir="fold_{}".format(fold_idx))#Sub_Logs Temp_Logs

    profiler = SimpleProfiler(dirpath=root/ "Logs" / exp_name / "version_{}".format(version) , filename="profilers")
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator = 'gpu', gpus=[gpu_idx], logger=logger, profiler=profiler)
    trainer.fit(model, train_loader, val_loader)
    return


if __name__ == '__main__':
    main()
