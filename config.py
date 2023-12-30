import argparse
import argparse

def parse_encoder(parser=None, disc=None):
    if parser is None:
        parser = argparse.ArgumentParser(description=disc)

    parser.add_argument('--root', type=str, default= "D2Match",
                        help='root of main folder')
    parser.add_argument('--gpu_idx', type=int, default=1,
                    help='Training GPU id')

    parser.add_argument('--num_layers', type=int, default=5,
                    help='Number of model layers')
    parser.add_argument('--hidden_dim', type=int, default=10,
                    help='Training hidden size')

    parser.add_argument('--fold_num', type=int, default=5,
                    help='Number of fold splits')
    parser.add_argument('--fold_idx', type=int, default=0,
                    help='Index of fold')
    
    parser.add_argument('--model_name', type=str, default="T",
                        help='Name of the model to be used')
    parser.add_argument('--dataset_name', type=str, default="syn",
                        help='Name of the dataset to be used')

    parser.add_argument('--no_feature', action='store_true', default=False,
                    help='Do not use node feature')

    parser.add_argument('--max_epoch', type=int, default=500,
                    help='max training epoch')

    parser.add_argument('--no_gnn_update', action='store_false', default=True,
                    help='Do not use gnn update module')
    
    parser.add_argument('--no_subtree_update', action='store_false', default=True,
                    help='Do not use subtree update module')

    parser.add_argument('--no_gnn_interact', action='store_false', default=True,
                    help='Do not use gnn-subtree interact module')

    parser.add_argument('--cc', action='store_true', default=False,
                    help='Use chordless cycle')

    parser.add_argument('--num_samples', type=int, default=5,
                    help='number of samples')

    parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name for logging")

    parser.add_argument("--version", type=int, default=-1,
                    help="Fix version of logs to fix the log direction so that model will skip the experiment to avoid repeat")

    parser.add_argument("--split_seed", type=int, default=-1,
                    help="random seed to permute the data for split validation")

    parser.add_argument("--batch_size", type=int, default=10,
                    help="batch_size")

    return parser