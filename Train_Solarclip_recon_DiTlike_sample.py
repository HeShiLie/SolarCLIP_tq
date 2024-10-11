# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image

from Model.diffusion import create_diffusion
from Model.models import get_DiT_model_from_args, get_VAE_model_from_args
from Data import Solardataloader_subset
from Data.utils import transfer_date_to_id
from Data.Solardataloader import enhance_funciton

import matplotlib.pyplot as plt
import argparse
import os
import json
import numpy as np

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP model.')

    parser.add_argument('--config_dir', type=str, 
                        default='None')
    parser.add_argument('--vae_config_dir', type=str,
                        default='/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/DiTlike/VAEpretrain/0094-0094/_0912_res_true_attn_flase/args.json', help='The path to the VAE model.')
    parser.add_argument('--vae_model_id', type=int,
                        default=240, help='The id of the VAE model.')
    parser.add_argument("--clip_model_path", type=str,
                        default="checkpoints/recon/0816_test1/SolarCLIP", help="The path to the SolarCLIP model.")
    parser.add_argument("--clip_model_id", type=int,
                        default=200, help="The id of the SolarCLIP model.")

    # Training parameters
    parser.add_argument('--global_batch_size', type=int,
                        default=90, help='Global batch size for training')
    parser.add_argument('--global_seed', type=int,
                        default=42, help='Global seed for training')
    parser.add_argument('--optimizer', type=str, 
                        default='AdamW', help='Optimizer for training')
    parser.add_argument('--learning_rate', type=float,
                        default=3e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, 
                        default=1000, help='Number of training epochs')
    parser.add_argument('--test_freq', type=int, 
                        default=100, help='Frequency of testing the model')
    parser.add_argument('--save_freq', type=int,
                        default=100, help='Frequency of saving the model')
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/DiTlike", help="The output path to save the model.")

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='Number of data loading workers')
    parser.add_argument('--machine', type=str, 
                        default='A40', help='Machine type for training')
    parser.add_argument('--sampler', type=str, 
                        default='distributed',help='Sampler for training')

    # Model parameters
    parser.add_argument('--input_size', type=int, 
                        default=64, help='Input size of the model')
    parser.add_argument('--patch_size', type=int, 
                        default=4, help='Patch size for the model')
    parser.add_argument('--in_channels', type=int, 
                        default=3, help='Number of input channels')
    parser.add_argument('--width', type=int, 
                        default=768, help='Width of the model')
    parser.add_argument('--norm_type', type=str, 
                        default='bn1d', help='Normalization type for the model')
    parser.add_argument('--depth', type=int, 
                        default=12, help='Depth of the model')
    parser.add_argument('--num_heads', type=int, 
                        default=48, help='Number of attention heads')
    parser.add_argument('--token_type', type=str, 
                        default='all embedding', help='Type of token used in the model')
    parser.add_argument('--layers', type=int, 
                        default=12, help='Number of layers in the model')
    parser.add_argument('--learn_sigma', type=bool,
                        default=True, help='Flag to indicate if sigma is learnable')
    parser.add_argument('--deprojection_type', type=str, 
                        default='linear', help='Type of deprojection used in the model')
    parser.add_argument('--with_bias', type=bool,
                        default=True, help='Flag to indicate if the model uses bias')

    # Modal parameters
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')
    parser.add_argument('--enhance_list', type=list, nargs="+", 
                        default=[['log1p', 1], ['log1p', 1]], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", 
                        default=[1024,0.5,90], help='Image preprocess list for training [resize, flip, rotate]')
    parser.add_argument('--weight_type_list', type=list, nargs="+", 
                        default=['cv-rdbu', '3sgm-continous'], help='Weight type list for training')
    parser.add_argument('--decoder_modal', type=str, 
                        default='0094-0094', help='Decoder modal for training')

    return parser.parse_args()

def load_args_from_json(args,config_dir):
    with open(f'{config_dir}', 'r') as f:
        arg_json = json.load(f)
    for arg in arg_json:
        setattr(args, arg, arg_json[arg])
    return args

def main(args, vae_args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda:3" if torch.cuda.is_available() else "cpu"

    clip_model_path = f"{args.clip_model_path}/epoch_{args.clip_model_id}.pt"
    SolarCLIP_dict = torch.load(clip_model_path)['model']
    visual_mag_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_mag.')}
    visual_mag_state_dict = {k[len('visual_mag.'):]: v for k, v in visual_mag_state_dict.items()}
    visual_0094_state_dict = {k: v for k, v in SolarCLIP_dict.items() if k.startswith('visual_H.')}
    visual_0094_state_dict = {k[len('visual_H.'):]: v for k, v in visual_0094_state_dict.items()}


    # Load model:
    latent_size = args.input_size 
    model = get_DiT_model_from_args(args).to(device)
    model.eval()
    model.load_state_dict(torch.load('/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/DiTlike/0094-0094/use_vae_lambda_1e-1_bn1d/model/epoch_1000.pt'
                                     , map_location=device)['model'])
    requires_grad(model, False)

    diffusion = create_diffusion(timestep_respacing="")

    vae = get_VAE_model_from_args(vae_args).to(device)
    vae.train()
    vae_checkpoint_path = os.path.dirname(args.vae_config_dir)
    vae_model_path = f"{vae_checkpoint_path}/model/epoch_{args.vae_model_id}.pt"
    vae.load_state_dict(torch.load(vae_model_path, map_location=device)['model'])
    requires_grad(vae, False)

    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12*10,time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list,  load_imgs= True, enhance_list=[args.image_preprocess[0],0,0], 
                                                           batch_size=5, shuffle=False, num_workers=args.num_workers
                                                    )

    with torch.no_grad():
        model.eval()
        # val_loss = []
        for i, data in enumerate(val_loader):
            data = data.to(device)
            if args.decoder_modal == 'magnet-magnet':
                x = data[:, 0, :, :, :]
                x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                model.y_vit.load_state_dict(visual_mag_state_dict)
                requires_grad(model.module.y_embedder, False)
                y = model.y_embedder(x)
            elif args.decoder_modal == 'magnet-0094': # todo
                pass
                # x = data[:, 0, :, :, :]
                # x = enhance_funciton(x, args.enhance_list[0][0], args.enhance_list[0][1])
                # model.y_embedder.load_state_dict(visual_0094_state_dict)
                # requires_grad(model.y_embedder, False)
                # y = model.y_embedder(x)
            elif args.decoder_modal == '0094-0094':
                x = data[:, 1, :, :, :]
                x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                model.y_vit.load_state_dict(visual_0094_state_dict)
                requires_grad(model.y_embedder, False)
                y = model.y_embedder(x)
            elif args.decoder_modal == '0094-magnet': # todo
                # x = data[:, 1, :, :, :]
                # x = enhance_funciton(x, args.enhance_list[1][0], args.enhance_list[1][1])
                # model.y_embedder.load_state_dict(visual_mag_state_dict)
                # requires_grad(model.y_embedder, False)
                # y = model.y_embedder(x)
                pass
            else:
                raise ValueError(f"Decoder modal {args.decoder_modal} not support.")
            z = torch.randn(5, args.in_channels, latent_size, latent_size, device=device)
            model_kwargs = dict(y=y)
            samples = diffusion.p_sample_loop(
        model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
            samples = vae.decode(samples)

            h_image = x.cpu().numpy()
            recon_image = samples.cpu().numpy()
            mag_image = data[:, 0, :, :, :]
            mag_image = enhance_funciton(mag_image, args.enhance_list[0][0], args.enhance_list[0][1])
            mag_image = mag_image.cpu().numpy()

            vmin_magnet = np.min(mag_image)
            vmax_magnet = np.max(mag_image)
            vmax_magnet = np.max([np.abs(vmin_magnet), np.abs(vmax_magnet)])/2
            vmin_magnet = -vmax_magnet
            vmin_0094 = 0
            vmax_0094 = np.max(h_image)

            print(vmin_magnet, vmax_magnet, vmin_0094, vmax_0094)
            
            plt.figure(figsize=(30, 12))
            for i in range(5):
                plt.subplot(3, 5, i+1)
                plt.imshow(h_image[i,0,:,:], cmap='Reds',vmin=vmin_0094, vmax=vmax_0094)
                plt.axis('off')
                plt.title('0094 original')

                plt.subplot(3, 5, i+6)
                plt.imshow(recon_image[i,0,:,:], cmap='Reds',vmin=vmin_0094, vmax=vmax_0094)
                plt.axis('off')
                plt.title('0094 recon')

                plt.subplot(3, 5, i+11)
                plt.imshow(mag_image[i,0,:,:], cmap='RdBu_r',vmin=vmin_magnet, vmax=vmax_magnet)
                plt.axis('off')
                plt.title('Magnet orginal')

            # Save and display images:
            plt.savefig('output/0918.jpg',bbox_inches='tight')
            plt.close()

            break


    

if __name__ == "__main__":
    args = parse_args()
    _ = args.config_dir
    if args.config_dir != 'None':
        args = load_args_from_json(args, args.config_dir)
    args.config_dir = _

    vae_args = argparse.ArgumentParser().parse_args()
    vae_args = load_args_from_json(vae_args, args.vae_config_dir)

    main(args, vae_args)
