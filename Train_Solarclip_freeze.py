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
from Model.get_weights import get_weights

import os
import random
import time

from Model.SolarCLIP_modify import get_model_from_args
from Model.vit import PretrainModel

random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SolarCLIP freeze model.')

    parser.add_argument('--config_dir', type=str, default='None')

    # Training parameters
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--test_freq', type=int, default=100,
                        help='Frequency of testing the model')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Frequency of saving the model')
    parser.add_argument('--inner_loss_rate', type=float, 
                        default=0, help='Inner loss rate for training')
    parser.add_argument('--device', type=str,
                        default='cuda:1', help='Device for training')

    # DataLoader parameters
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--embed_dim', type=int,
                        default=768, help='Embedding dimension')
    parser.add_argument('--vision_width', type=int, 
                        default=768,help='Width of the vision transformer')
    parser.add_argument('--image_resolution_mag', type=int,
                        default=1024, help='Resolution of the mag image')
    parser.add_argument('--vision_layers_mag', type=int, default=12,
                        help='Number of layers in mag vision transformer')
    parser.add_argument('--vision_patch_size_mag', type=int,
                        default=64, help='Patch size for mag vision transformer')
    parser.add_argument('--image_resolution_H', type=int,
                        default=1024, help='Resolution of the H image')
    parser.add_argument('--vision_layers_H', type=int, default=12,
                        help='Number of layers in H vision transformer')
    parser.add_argument('--vision_patch_size_H', type=int,
                        default=64, help='Patch size for H vision transformer')
    parser.add_argument('--token_type', type=str,
                        default='all embedding', help='Token type for CLIP model')
    parser.add_argument('--weight_type_list', type=str, nargs="+", default=['cv-rdbu', '3sgm-continous'], help='Weight type list for training')

    # Modal parameters
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')
    parser.add_argument('--enhance_list', type=list, nargs="+", default=[
                        ['log1p', 1], ['log1p', 1]], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", default=[1024,0.5,90], help='Image preprocess list for training [resize, flip, rotate]')
    parser.add_argument('--embed_state_root', type=str, default='checkpoints/pretrain/endecoder_distinct/modal_magnet_0094/enhance_log1p_1/preprocess_1024_0.5_90/weights_typecv-rdbu_3sgm-continous', help='Embedding state root path')
    parser.add_argument('--embed_model1_id', type=int, required=True, help='select the model id for the first modal')
    parser.add_argument('--embed_model2_id', type=int, required=True, help='select the model id for the second modal')
    parser.add_argument("--checkpoint_path", type=str,
                        default="/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/SolarCLIP_freeze/", help="The output path to save the model.")

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

    checkpoint_path = args.checkpoint_path
    checkpoint_path += 'endecoder_distinct/'
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
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                             start_date, end_date], modal_list=args.modal_list, load_imgs= True, enhance_list=args.image_preprocess, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                           start_date, end_date], modal_list=args.modal_list,  load_imgs= True,enhance_list=[args.image_preprocess[0],0,0], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    SolarModel = get_model_from_args(args).to(device)

    encoder_decoder = PretrainModel(embedding_dim=args.embed_dim, input_size=args.image_resolution_mag, patch_size=args.vision_patch_size_mag, output_dim=1, output_size=args.image_resolution_mag, dropout_prob=0.1, deprojection_type='linear', with_bias=True).to(device)
    encoder_1_dir = f'{args.embed_state_root}/{args.modal_list[0]}/model/epoch_{args.embed_model1_id}.pt'
    encoder_1_state_dict = torch.load(encoder_1_dir, map_location=torch.device('cpu'))
    encoder_decoder.load_state_dict(encoder_1_state_dict['model'])
    encoder_decoder.to(device)
    SolarModel.visual_mag.conv1.load_state_dict(encoder_decoder.encoder.state_dict())

    # encoder_2_dir = f'{args.embed_state_root}/{args.modal_list[1]}/model/epoch_{args.embed_model2_id}.pt'
    # encoder_2_state_dict = torch.load(encoder_2_dir, map_location=torch.device('cpu'))
    # encoder_decoder.load_state_dict(encoder_2_state_dict['model'])
    # encoder_decoder.to(device)
    SolarModel.visual_H.conv1.load_state_dict(encoder_decoder.encoder.state_dict())
    del encoder_decoder


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
        for param in SolarModel.visual_mag.conv1.parameters():
            param.requires_grad = False
        for param in SolarModel.visual_H.conv1.parameters():
            param.requires_grad = False
        acc_results = []
        epoch_time = time.time()
        for i, data in enumerate(train_loader):

            data = data.to(device)
            modal_1 = data[:, 0, :, :, :]
            modal_2 = data[:, 1, :, :, :]
            modal_1 = enhance_funciton(modal_1, args.enhance_list[0][0], args.enhance_list[0][1])
            modal_2 = enhance_funciton(modal_2, args.enhance_list[1][0], args.enhance_list[1][1])
            _, weight_1 = get_weights(args.weight_type_list[0], modal_1) # B, (n_h*n_w)
            _, weight_2 = get_weights(args.weight_type_list[1], modal_2) # B, (n_h*n_w)
            iteration_txt = f"Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            loss, loss_inner, acc, _, _ = SolarModel.calculate_loss(modal_1, modal_2, args.inner_loss_rate, weight_1, weight_2)
            iteration_txt += f" Forward time: {
                (time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer.zero_grad()
            loss.backward()
            if args.inner_loss_rate > 0:
                loss_inner*=args.inner_loss_rate
                loss_inner.backward()
            optimizer.step()
            iteration_txt += f" Backward time: {
                (time.time()-epoch_time)/60:.2f} min |"
            print(iteration_txt)
            epoch_time = time.time()

            logger_train_loss.append(loss.item())
            logger_train_loss_inner.append(loss_inner.item())
            logger_train_acc.append(acc)
            logger_lr.append(scheduler.get_last_lr()[0])
            acc_results.append(acc)
            print('acc:',np.mean(acc_results))

        scheduler.step()

        print(f"epoch {epoch+1:<6}: Loss = {loss.item():<7.8f} | Loss_inner = {loss_inner.item():<7.8f} | acc = {np.mean(acc_results):<7.2%} | Lr rate = {
              scheduler.get_last_lr()[0]:<7.8f} | Time = {(time.time()-start_time)/60:.2f} min")
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
                    _, weight_1 = get_weights(args.weight_type_list[0], modal_1) # B, (n_h*n_w)
                    _, weight_2 = get_weights(args.weight_type_list[1], modal_2) # B, (n_h*n_w)

                    loss, loss_inner, acc, _, _ = SolarModel.calculate_loss(modal_1, modal_2, args.inner_loss_rate, weight_1, weight_2)
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
