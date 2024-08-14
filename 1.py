import torch
import numpy as np
import argparse
import json
import pickle
from types import SimpleNamespace
from astropy.io import fits
#import clip
#import clip.model

from Data import Solardataloader_subset

import os
import random
import time

from Model.SolarCLIP import get_model_from_args

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP model.')

    parser.add_argument('--config_dir', type=str, default='None')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--test_freq', type=int, default=10, help='Frequency of testing the model')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving the model')

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--vision_width', type=int, default=768, help='Width of the vision transformer')
    parser.add_argument('--image_resolution_mag', type=int, default=224, help='Resolution of the mag image')
    parser.add_argument('--vision_layers_mag', type=int, default=12, help='Number of layers in mag vision transformer')
    parser.add_argument('--vision_patch_size_mag', type=int, default=32, help='Patch size for mag vision transformer')
    parser.add_argument('--image_resolution_H', type=int, default=224, help='Resolution of the H image')
    parser.add_argument('--vision_layers_H', type=int, default=12, help='Number of layers in H vision transformer')
    parser.add_argument('--vision_patch_size_H', type=int, default=32, help='Patch size for H vision transformer')
    parser.add_argument('--token_type', type=str, default='all embedding', help='Token type for CLIP model')

    #Modal parameters
    parser.add_argument('--modal_list', type=str, nargs = "+", default=['magnet','0094'], help='Modal list for training')
    parser.add_argument('--enhance_list', type=list, nargs = "+", default=[['log1p',224,1],['log1p',224,1]], help='Enhance list for training')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')
    parser.add_argument("--checkpoint_path", type=str, default = "/mnt/nas/home/huxing/202407/ctf/SolarCLIP/checkpoints/", help="The output path to save the model.")

    return parser.parse_args()

def save_args(args, checkpoint_dir):
    with open(f'{checkpoint_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f'args saved to {checkpoint_dir}/args.json')

def load_args(config_dir):
    with open(f'{config_dir}/args.json', 'r') as f:
        arg_json = json.load(f)
    arg_json = SimpleNamespace(**arg_json)
    return arg_json

if __name__ == '__main__':
    import argparse
    from types import SimpleNamespace
    import json

    def parse_args():
        parser = argparse.ArgumentParser(description='Train SolarCLIP model.')

        parser.add_argument('--config_dir', type=str, default='None')

        # Training parameters
        parser.add_argument('--batch_size', type=int,
                            default=256, help='Batch size for training')
        parser.add_argument('--learning_rate', type=float,
                            default=1e-1, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs')
        parser.add_argument('--test_freq', type=int, default=10,
                            help='Frequency of testing the model')
        parser.add_argument('--save_freq', type=int, default=10,
                            help='Frequency of saving the model')
        parser.add_argument('--inner_loss_rate', type=float, default=0)

        # DataLoader parameters
        parser.add_argument('--num_workers', type=int, default=8,
                            help='Number of data loading workers')

        # Model parameters
        parser.add_argument('--embed_dim', type=int,
                            default=512, help='Embedding dimension')
        parser.add_argument('--vision_width', type=int, default=768,
                            help='Width of the vision transformer')
        parser.add_argument('--image_resolution_mag', type=int,
                            default=224, help='Resolution of the mag image')
        parser.add_argument('--vision_layers_mag', type=int, default=12,
                            help='Number of layers in mag vision transformer')
        parser.add_argument('--vision_patch_size_mag', type=int,
                            default=32, help='Patch size for mag vision transformer')
        parser.add_argument('--image_resolution_H', type=int,
                            default=224, help='Resolution of the H image')
        parser.add_argument('--vision_layers_H', type=int, default=12,
                            help='Number of layers in H vision transformer')
        parser.add_argument('--vision_patch_size_H', type=int,
                            default=32, help='Patch size for H vision transformer')
        parser.add_argument('--token_type', type=str,
                            default='all embedding', help='Token type for CLIP model')

        # Modal parameters
        parser.add_argument('--modal_list', type=str, nargs="+",
                            default=['magnet', '0094'], help='Modal list for training')
        parser.add_argument('--enhance_list', type=list, nargs="+", default=[
                            ['log1p', 224, 1], ['log1p', 224, 1]], help='Enhance list for training')
        parser.add_argument('--device', type=str,
                            default='cuda:0', help='Device for training')
        parser.add_argument("--checkpoint_path", type=str,
                            default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP/checkpoints/", help="The output path to save the model.")

        return parser.parse_args()


    def save_args(args, checkpoint_dir):
        with open(f'{checkpoint_dir}/args.json', 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f'args saved to {checkpoint_dir}/args.json')


    def load_args(config_dir):
        with open(f'{config_dir}', 'r') as f:
            arg_json = json.load(f)
        arg_json = SimpleNamespace(**arg_json)
        return arg_json

    args = parse_args()
    args_ = load_args('./configs/args1.json')
    print(args_)
    for i in vars(args_):
        args.__setattr__(i, vars(args_)[i])

    print(args)
