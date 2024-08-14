import torch
import numpy as np
import argparse
import json
import pickle
from types import SimpleNamespace

import os
import random
import time

from Data import Solardataloader_subset
from Data.utils import transfer_date_to_id
from Data.Solardataloader import enhance_funciton

from Model.SolarCLIP_embedfreeze import get_model_from_args
from Solarclip_test import calculate_loss

random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP model.')

    parser.add_argument('--config_dir', type=str, default='None')

    # Training parameters
    parser.add_argument('--batch_size', type=int,
                        default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--test_freq', type=int, default=10,
                        help='Frequency of testing the model')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving the model')
    parser.add_argument('--inner_loss_rate', type=float, default=0)

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--train_date_list', type=list, nargs="+",
                         default=[[2010,5,1],[2020,6,30]], help='Date of the train data')
    parser.add_argument('--test_date_list', type=list, nargs="+",
                         default=[[2020,6,30],[2024,6,30]], help='Date of the test data')
    parser.add_argument('--time_step', type=int, default=10,
                        help='Time step of the data')
    parser.add_argument('--load_images', type=bool, default=True,
                        help='Load images in memory or not')

    # Model parameters
    parser.add_argument('--vision_width', type=int, default=768,
                        help='Width of the vision transformer, the output dimension of the model')
    parser.add_argument('--image_resolution_mag', type=int,
                        default=224, help='Resolution of the mag image')
    parser.add_argument('--vision_patch_size_mag', type=int,
                        default=32, help='Patch size for mag vision transformer')
    parser.add_argument('--hidden_dropout_prob_embeddings_mag', type=float,
                        default=1e-1, help='Dropout rate for embeddings in mag vision transformer')
    parser.add_argument('--vision_layers_mag', type=int, default=12,
                        help='Number of layers in mag vision transformer')
    parser.add_argument('--hidden_dropout_prob_transformer_mag', type=float,
                        default=1e-1, help='Dropout rate for transformer in mag vision transformer')
    parser.add_argument('--image_resolution_H', type=int,
                        default=224, help='Resolution of the H image')
    parser.add_argument('--vision_patch_size_H', type=int,
                        default=32, help='Patch size for H vision transformer')
    parser.add_argument('--hidden_dropout_prob_embeddings_H', type=float,
                        default=0.1, help='Dropout rate for embeddings in H vision transformer')
    parser.add_argument('--vision_layers_H', type=int, default=12,
                        help='Number of layers in H vision transformer')
    parser.add_argument('--hidden_dropout_prob_transformer_H', type=float,
                        default=1e-1, help='Dropout rate for transformer in H vision transformer')
    parser.add_argument('--token_type', type=str,
                        default='all embedding', help='Token type for CLIP model')
    parser.add_argument('--embed_mag_pt', type=str,
                        default='None', help='Pretrained model path of magnetogram')
    parser.add_argument('--embed_0094_pt', type=str,
                        default='None', help='Pretrained model path of 0094')

    # Modal parameters
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')
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

    args = parse_args()
    _ = args.config_dir
    if args.config_dir != 'None':
        args = load_args_from_json(args, args.config_dir)
    args.config_dir = _
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")

    checkpoint_path = args.checkpoint_path + 'embedfreeze/'
    checkpoint_path += args.token_type + '/'
    for i in range(len(args.modal_list)):
        checkpoint_path += args.modal_list[i] + '_'
        for j in range(len(args.enhance_list[i])):
            checkpoint_path += str(args.enhance_list[i][j]) + '_'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model_checkpoint_path = checkpoint_path + '/model/'
    logger_checkpoint_path = checkpoint_path + '/logger/'
    if not os.path.exists(f'{model_checkpoint_path}'):
        os.makedirs(f'{model_checkpoint_path}')
    if not os.path.exists(f'{logger_checkpoint_path}'):
        os.makedirs(f'{logger_checkpoint_path}')

    save_args(args, checkpoint_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('device:', device)

    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=args.time_step,time_interval=[
                                                             start_date, end_date], modal_list=args.modal_list, load_imgs= args.load_images, enhance_list=args.image_preprocess, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=args.time_step,time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list,  load_imgs= args.load_images, enhance_list=[args.image_preprocess[0],0,0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    SolarModel = get_model_from_args(args).to(device)

    Pretrained_dict_mag = torch.load(args.embed_mag_pt, map_location=device)
    encoder_state_dict = {k: v for k, v in Pretrained_dict_mag['model'].items() if k.startswith('encoder.')}
    encoder_state_dict = {k[len('encoder.'):]: v for k, v in encoder_state_dict.items()}
    SolarModel.visual_mag.Embeddings.load_state_dict(encoder_state_dict)
    for param in SolarModel.visual_mag.Embeddings.parameters():
        param.requires_grad = False
    
    Pretrained_dict_0094 = torch.load(args.embed_0094_pt, map_location=device) 
    encoder_state_dict = {k: v for k, v in Pretrained_dict_0094['model'].items() if k.startswith('encoder.')}
    encoder_state_dict = {k[len('encoder.'):]: v for k, v in encoder_state_dict.items()}
    SolarModel.visual_mag.Embeddings.load_state_dict(encoder_state_dict)
    for param in SolarModel.visual_mag.Embeddings.parameters():
        param.requires_grad = False


    optimizer = torch.optim.SGD(
        SolarModel.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    epochs = args.epochs
    test_epoch = epochs//args.test_freq
    save_epoch = epochs//args.save_freq

    logger_train_loss = []
    logger_train_loss_inner = []
    logger_train_acc = []
    logger_lr = []
    logger_val_loss = []
    logger_val_loss_inner = []
    logger_val_acc = []

    start_time = time.time()
    print('Start training')
    for epoch in range(epochs):

        SolarModel.train()
        acc_results = []
        epoch_time = time.time()
        for i, data in enumerate(train_loader):

            data = data.to(device)
            modal_1 = data[:, 0, :, :, :]
            modal_2 = data[:, 1, :, :, :]
            modal_1 = enhance_funciton(modal_1, args.enhance_list[0][0], args.enhance_list[0][1])
            modal_2 = enhance_funciton(modal_2, args.enhance_list[1][0], args.enhance_list[1][1])
            data = torch.stack([modal_1, modal_2], dim=1)
            iteration_txt = f"Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()
            loss, loss_inner, acc, _, _ = calculate_loss(
                SolarModel, data, args.inner_loss_rate, criterion=torch.nn.functional.cross_entropy)
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer.zero_grad()
            loss_inner*=args.inner_loss_rate
            loss+=loss_inner
            loss.backward()
            optimizer.step()
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.2f} min |"
            print(iteration_txt)
            epoch_time = time.time()

            logger_train_loss.append(loss.item())
            logger_train_loss_inner.append(loss_inner.item())
            logger_train_acc.append(acc)
            logger_lr.append(scheduler.get_last_lr()[0])
            acc_results.append(acc)
            print('mean:',np.mean(acc_results))


        scheduler.step()

        print(f"epoch {epoch+1:<6}: Loss = {loss.item():<7.8f} | Loss_inner = {loss_inner.item():<7.8f} | acc = {np.mean(acc_results):<7.2%} | Lr rate = {scheduler.get_last_lr()[0]:<7.8f} | Time = {(time.time()-start_time)/60:.2f} min")
        start_time = time.time()

        if (epoch+1) % test_epoch == 0:
            with torch.no_grad():
                SolarModel.eval()
                loss_results = []
                loss_inner_results = []
                acc_results = []
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    modal_1 = data[:, 0, :, :, :]
                    modal_2 = data[:, 1, :, :, :]
                    modal_1 = enhance_funciton(modal_1, args.enhance_list[0][0], args.enhance_list[0][1])
                    modal_2 = enhance_funciton(modal_2, args.enhance_list[1][0], args.enhance_list[1][1])
                    data = torch.stack([modal_1, modal_2], dim=1)
                    loss, loss_inner, acc, _, _ = calculate_loss(
                        SolarModel, data, inner_loss_rate=args.inner_loss_rate, criterion=torch.nn.functional.cross_entropy)
                    loss_results.append(loss.item())
                    loss_inner_results.append(loss_inner.item())
                    acc_results.append(acc)

                logger_val_loss.append(np.mean(loss_results))
                logger_val_loss_inner.append(np.mean(loss_inner_results))
                logger_val_acc.append(np.mean(acc_results))

                result_txt = f'Epoch {epoch+1:>6}/{epochs:<6} | '

                result_txt += f'Train loss {logger_train_loss[-1]:<10.8f} | '
                result_txt += f'Train loss inner {logger_train_loss_inner[-1]:<10.8f} | '
                result_txt += f'Train acc {logger_train_acc[-1]:<10.2%} | '
                result_txt += f'Val loss {logger_val_loss[-1]:<10.8f} | '
                result_txt += f'Val loss inner {logger_val_loss_inner[-1]:<10.8f} | '
                result_txt += f'Val acc {logger_val_acc[-1]:<10.2%} | '
                result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
                print(result_txt)
                start_time = time.time()

                with open(f'{logger_checkpoint_path}logger_train_loss.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss, f)
                with open(f'{logger_checkpoint_path}logger_train_loss_inner.pkl', 'wb') as f:
                    pickle.dump(logger_train_loss_inner, f)
                with open(f'{logger_checkpoint_path}logger_train_acc.pkl', 'wb') as f:
                    pickle.dump(logger_train_acc, f)
                with open(f'{logger_checkpoint_path}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr, f)
                with open(f'{logger_checkpoint_path}logger_val_loss.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss, f)
                with open(f'{logger_checkpoint_path}logger_val_loss_inner.pkl', 'wb') as f:
                    pickle.dump(logger_val_loss_inner, f)
                with open(f'{logger_checkpoint_path}logger_val_acc.pkl', 'wb') as f:
                    pickle.dump(logger_val_acc, f)

        if (epoch+1) % save_epoch == 0:
            torch.save({'model': SolarModel.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch},
                       f'{model_checkpoint_path}epoch_{epoch+1}.pt')
            print(f'Model saved {(epoch+1)/epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')
            start_time = time.time()

    # torch.save(RSModel.state_dict(), './weight/rsclip_model.pth')


if __name__ == '__main__':
    main()
