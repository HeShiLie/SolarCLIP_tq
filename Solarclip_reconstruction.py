import torch
import numpy as np
import argparse
import json
import pickle
from types import SimpleNamespace
# import clip
# import clip.model

from Data import Solardataloader_subset
from Data.utils import transfer_date_to_id
from Data.Solardataloader import enhance_funciton

import os
import random
import time

from Model.SolarCLIP import get_model_from_args
from Model.decoder import LinearDecoder, get_decoder_from_args
from Solarclip_test import calculate_loss, calculate_loss_reconstruction

random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP decoder.')

    parser.add_argument('--decoder_config_dir', type=str, default='None')
    parser.add_argument('--encoder_config_dir', type=str, default='None')

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

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--token_type', type=str,
                        default='all embedding', help='Token type for CLIP model')
    parser.add_argument('--embed_dim',type=int,
                        default=512,help='Embedding dimension')
    parser.add_argument('--out_dim',type=int,
                        default=1024,help='LinearDecoder Output dimension')
    parser.add_argument('--hidden_dim_1',type=int,
                        default=1024,help='LinearDecoder hidden dimension 1')
    parser.add_argument('--hidden_dim_2',type=int,
                        default=1024,help='LinearDecoder hidden dimension 2')
    parser.add_argument('--hidden_dim_3',type=int,
                        default=1024,help='LinearDecoder hidden dimension 3')

    # Modal parameters
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')
    parser.add_argument('--decode_modal', type=str,
                        default='magnet', help='Modal for decoding')
    parser.add_argument('--enhance_list', type=list, nargs="+", default=[
                        ['log1p', 1], ['log1p', 1]], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", default=[224,0.5,90], help='Image preprocess list for training [resize, flip, rotate]')
    parser.add_argument('--device', type=str,
                        default='cuda:0', help='Device for training')
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP/checkpoints/", help="The output path to save the model.")

    return parser.parse_args()


def save_args(args, checkpoint_dir):
    with open(f'{checkpoint_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f'args saved to {checkpoint_dir}/args.json')


def load_args_from_json(args,config_dir):
    with open(f'{config_dir}', 'r') as f:
        arg_json = json.load(f)
    for arg in arg_json:
        setattr(args, arg, arg_json[arg])
    return args


def main():
    args_encoder = SimpleNamespace()
    args = parse_args()
    _ = args.encoder_config_dir
    if args.encoder_config_dir != 'None':
        args_encoder = load_args_from_json(args_encoder, args.encoder_config_dir)  
    # args.config_dir = _
    print('Encoder config:')
    for arg in vars(args_encoder):
        print(f"{arg:<30}: {getattr(args_encoder, arg)}")

    if args.decoder_config_dir != 'None':
        args = load_args_from_json(args, args.decoder_config_dir)
    args.decoder_config_dir = _
    print('Decoder config:')
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")


    checkpoint_path = args.checkpoint_path
    checkpoint_path += args.token_type + '/'
    for i in range(len(args.modal_list)):
        checkpoint_path += args.modal_list[i] + '_'
        for j in range(len(args.enhance_list[i])):
            checkpoint_path += str(args.enhance_list[i][j]) + '_'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    decoder_checkpoint_path = checkpoint_path + '/decoder/'
    decoder_checkpoint_path += args.decode_modal
    logger_checkpoint_path = decoder_checkpoint_path + '/logger/'
    decoder_model_checkpoint_path = decoder_checkpoint_path + '/model/'
    if not os.path.exists(f'{decoder_checkpoint_path}'):
        os.makedirs(f'{decoder_checkpoint_path}')
    if not os.path.exists(f'{logger_checkpoint_path}'):
        os.makedirs(f'{logger_checkpoint_path}')
    if not os.path.exists(f'{decoder_model_checkpoint_path}'):
        os.makedirs(f'{decoder_model_checkpoint_path}')

    save_args(args, decoder_checkpoint_path)


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('device:', device)

    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_interval=[
                                                             start_date, end_date], modal_list=args.modal_list, load_imgs = False, enhance_list=args.image_preprocess, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list, load_imgs = False, enhance_list=[args.image_preprocess[0],0,0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    encoder_checkpoint_path = args_encoder.checkpoint_path
    encoder_checkpoint_path += args_encoder.token_type + '/'
    for i in range(len(args_encoder.modal_list)):
        encoder_checkpoint_path += args_encoder.modal_list[i] + '_'
        for j in range(len(args_encoder.enhance_list[i])):
            encoder_checkpoint_path += str(args_encoder.enhance_list[i][j]) + '_'
    print(encoder_checkpoint_path)

    if args.load_id is None:    #todo
        load_id = args_encoder.epochs
        for i in range(21):
            _ = f'{encoder_checkpoint_path}/model/epoch_{i}.pt'
            if os.path.exists(_):
                load_encoder_dir = _
        assert load_encoder_dir is not None
    else:
        load_encoder_dir = f'{encoder_checkpoint_path}/model/epoch_{args.load_id}.pt'

    SolarModel = get_model_from_args(args_encoder).to(device)
    SolarModel.load_state_dict(torch.load(load_encoder_dir)['model'])
    SolarModel = SolarModel.to(device)
    for param in SolarModel.parameters():
        param.requires_grad = False
    
    Decoder = get_decoder_from_args(args).to(device)  #todo
    optimizer_decoder = torch.optim.SGD(
        Decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_decoder, T_max=args.epochs)

    epochs = args.epochs
    test_epoch = epochs//args.test_freq
    save_epoch = epochs//args.save_freq

    logger_train_loss = []
    logger_train_PSNR = []
    logger_lr = []
    logger_val_loss = []
    logger_val_PSNR = []

    start_time = time.time()
    print('Start training')
    for epoch in range(epochs):

        Decoder.train()
        PSNR_results = []
        epoch_time = time.time()
        for i, data in enumerate(train_loader):

            data = data.to(device)
            modal_1 = data[:, 0, :, :, :]
            modal_2 = data[:, 1, :, :, :]
            modal_1 = enhance_funciton(modal_1, args.enhance_list[0][0], args.enhance_list[0][1])
            modal_2 = enhance_funciton(modal_2, args.enhance_list[1][0], args.enhance_list[1][1])
            data = torch.stack([modal_1, modal_2], dim=1)
            iteration_txt = f"Iteration {i} | Data time: {
                (time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()
            
            loss, psnr = calculate_loss_reconstruction(args.decode_modal,Decoder, data, SolarModel)

            iteration_txt += f" Forward time: {
                (time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()
            iteration_txt += f" Backward time: {
                (time.time()-epoch_time)/60:.2f} min |"
            print(iteration_txt)
            epoch_time = time.time()

            logger_train_loss.append(loss.item())
            logger_train_PSNR.append(psnr)
            logger_lr.append(scheduler_decoder.get_last_lr()[0])
            PSNR_results.append(psnr)
        scheduler_decoder.step()
        
        print(f"epoch {epoch+1:<6}: Loss = {loss.item():<7.8f} | PSNR = {np.mean(PSNR_results):<7.2%} | Lr rate = {
              scheduler_decoder.get_last_lr()[0]:<7.8f} | Time = {(time.time()-start_time)/60:.2f} min")
        start_time = time.time()

        if (epoch+1) % test_epoch == 0:
            with torch.no_grad():
                SolarModel.eval()
                loss_results = []
                PSNR_results = []
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    modal_1 = data[:, 0, :, :, :]
                    modal_2 = data[:, 1, :, :, :]
                    modal_1 = enhance_funciton(modal_1, args.enhance_list[0][0], args.enhance_list[0][1])
                    modal_2 = enhance_funciton(modal_2, args.enhance_list[1][0], args.enhance_list[1][1])
                    data = torch.stack([modal_1, modal_2], dim=1)

                    loss, psnr = calculate_loss_reconstruction(args.decode_modal,Decoder, data, SolarModel)

                    loss_results.append(loss.item())
                    PSNR_results.append(psnr)

                logger_val_loss.append(np.mean(loss_results))
                logger_val_PSNR.append(np.mean(PSNR_results))

                result_txt = f'Epoch {epoch+1:>6}/{epochs:<6} | '

                result_txt += f'Train loss {logger_train_loss[-1]:<10.8f} | '
                result_txt += f'Train PSNR {logger_train_PSNR[-1]:<10.2%} | '
                result_txt += f'Val loss {logger_val_loss[-1]:<10.8f} | '
                result_txt += f'Val PSNR {logger_val_PSNR[-1]:<10.2%} | '
                result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
                print(result_txt)
                start_time = time.time()

                with open(f'{logger_checkpoint_path}logger_train_loss.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss, f)
                with open(f'{logger_checkpoint_path}logger_train_PSNR.pkl', 'wb') as f:
                    pickle.dump(logger_train_PSNR, f)
                with open(f'{logger_checkpoint_path}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr, f)
                with open(f'{logger_checkpoint_path}logger_val_loss.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss, f)
                with open(f'{logger_checkpoint_path}logger_val_PSNR.pkl', 'wb') as f:
                    pickle.dump(logger_val_PSNR, f)

        if (epoch+1) % save_epoch == 0:
            torch.save({'model': Decoder.state_dict(), 'optimizer': optimizer_decoder.state_dict(),
                        'scheduler': scheduler_decoder.state_dict(), 'epoch': epoch},
                       f'{decoder_model_checkpoint_path}epoch_{epoch+1}.pt')
            print(f'Model saved {(epoch+1)/epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')
            start_time = time.time()

    # torch.save(RSModel.state_dict(), './weight/rsclip_model.pth')


if __name__ == '__main__':
    main()
