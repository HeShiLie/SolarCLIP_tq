import torch
import numpy as np
import pickle
import argparse
import json

import os
import time

from Model.vit import PretrainModel
from Model.get_weights import get_weights

from Data.Solardataloader import enhance_funciton
from Data.utils import transfer_date_to_id
import Data.Solardataloader_subset as Solardataloader_subset

def load_args_from_json(args,config_dir):
    with open(f'{config_dir}', 'r') as f:
        arg_json = json.load(f)
    for arg in arg_json:
        setattr(args, arg, arg_json[arg])
    return args

def save_args(args, checkpoint_dir):
    with open(f'{checkpoint_dir}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f'args saved to {checkpoint_dir}/args.json')

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain encoder and decoder SolarCLIP model.')

    # Config Parameters
    parser.add_argument('--config_path', type=str, default='None' #todo
                        , help='Path to config file')
    
    # Training Parameters
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Batch size')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='Optimizer for training')
    parser.add_argument('--lr', type=float,
                        default=4e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, 
                        default=100, help='Number of epochs')
    parser.add_argument('--test_freq', type=int,
                        default=10, help='Test frequency')
    parser.add_argument('--save_freq', type=int,
                        default=10, help='Save frequency')
    parser.add_argument('--device', type=str,
                        default='cuda:0', help='Device for training')


    # Model Parameters
    parser.add_argument('--input_size', type=int, 
                        default=1024, help='Input size of the model')
    parser.add_argument('--embedding_dim', type=int, 
                        default=768, help='Embedding dimension of the model')
    parser.add_argument('--input_dim', type=int, 
                        default=1, help='Input dimension of the model')
    parser.add_argument('--patch_size', type=int, 
                        default=64, help='Patch size of the model')
    parser.add_argument('--dropout_prob', type=float, 
                        default=0.1, help='Dropout probability of the model')
    parser.add_argument('--output_dim', type=int, 
                        default=1, help='Output dimension of the model')
    parser.add_argument('--output_size', type=int, 
                        default=1024, help='Output size of the model')
    parser.add_argument('--deprojection_type', type=str, 
                        default='linear', help='Deprojection type of the model')
    parser.add_argument('--with_bias', type=bool, 
                        default=True, help='With bias of the model')
    
    # Modal Parameters
    parser.add_argument('--modal_list', type=str, nargs="+",
                        default=['magnet', '0094'], help='Modal list for training')
    parser.add_argument('--enhance_list', type=list, nargs="+", 
                        default=['log1p', 1], help='Enhance list for training')
    parser.add_argument('--image_preprocess', type=list, nargs="+", 
                        default=[1024, 0.5, 90], help='Image preprocess for training')
    parser.add_argument('--weights_type_list', type=str, nargs="+",
                        default=['cv-rdbu', '3sgm-continous'], help='Weights type list for training')
    
    # Save Parameters
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/pretrain/')
    parser.add_argument('--log_path', type=str, default='/mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/log/pretrain/')
    return parser.parse_args()

def train(args, train_loader, val_loader):

    pretrainModel_mag = PretrainModel(input_size=args.input_size, embedding_dim=args.embedding_dim, input_dim=args.input_dim, patch_size=args.patch_size, dropout_prob=args.dropout_prob, output_dim=args.output_dim, output_size=args.output_size, deprojection_type=args.deprojection_type, with_bias=args.with_bias)
    pretrainModel_mag = pretrainModel_mag.to(args.device)
    pretrainModel_0094 = PretrainModel(input_size=args.input_size, embedding_dim=args.embedding_dim, input_dim=args.input_dim, patch_size=args.patch_size, dropout_prob=args.dropout_prob, output_dim=args.output_dim, output_size=args.output_size, deprojection_type=args.deprojection_type, with_bias=args.with_bias)
    pretrainModel_0094 = pretrainModel_0094.to(args.device)

    checkpoint_path = args.checkpoint_path
    checkpoint_path += 'modal_' + str(args.modal_list[0]) + '_' + str(args.modal_list[1]) + '/'
    checkpoint_path += 'enhance_' + str(args.enhance_list[0]) + '_' + str(args.enhance_list[1]) + '/'
    checkpoint_path += 'preprocess_' + str(args.image_preprocess[0]) + '_' + str(args.image_preprocess[1]) + '_' + str(args.image_preprocess[2]) + '/'
    checkpoint_path += 'weights_type' + str(args.weights_type_list[0]) + '_' + str(args.weights_type_list[1]) 
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # /mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/pretrain/modal_magnet_0094/enhance_log1p_1/preprocess_1024_0.5_90/weights_typecv-rdbu_3sgm-continous/
    save_args(args, checkpoint_path)

    checkpoint_path_mag = checkpoint_path + '/magnet/'
    logger_checkpoint_path_mag = checkpoint_path_mag + 'logger/'
    model_checkpoint_path_mag = checkpoint_path_mag + 'model/'
    if not os.path.exists(checkpoint_path_mag):
        os.makedirs(checkpoint_path_mag)
    if not os.path.exists(f'{model_checkpoint_path_mag}'):
        os.makedirs(f'{model_checkpoint_path_mag}')
    if not os.path.exists(f'{logger_checkpoint_path_mag}'):
        os.makedirs(f'{logger_checkpoint_path_mag}')

    checkpoint_path_0094 = checkpoint_path + '/0094/'
    logger_checkpoint_path_0094 = checkpoint_path_0094 + 'logger/'
    model_checkpoint_path_0094 = checkpoint_path_0094 + 'model/'
    if not os.path.exists(checkpoint_path_0094):
        os.makedirs(checkpoint_path_0094)
    if not os.path.exists(f'{model_checkpoint_path_0094}'):
        os.makedirs(f'{model_checkpoint_path_0094}')
    if not os.path.exists(f'{logger_checkpoint_path_0094}'):
        os.makedirs(f'{logger_checkpoint_path_0094}')

    if args.optimizer == 'AdamW':
        optimizer_mag = torch.optim.AdamW(
            pretrainModel_mag.parameters(), lr=args.lr, weight_decay=1e-4)
        optimizer_0094 = torch.optim.AdamW(
            pretrainModel_0094.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'SGD':
        optimizer_mag = torch.optim.SGD(
            pretrainModel_mag.parameters(), lr=args.lr, weight_decay=1e-4)
        optimizer_0094 = torch.optim.SGD(
            pretrainModel_0094.parameters(), lr=args.lr, weight_decay=1e-4)
        
    scheduler_mag = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_mag, T_max=args.epochs)
    scheduler_0094 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_0094, T_max=args.epochs)

    test_epoch = args.epochs//args.test_freq
    save_epoch = args.epochs//args.save_freq

    logger_train_mse_mag = []
    logger_train_weighted_mse_mag = []  
    logger_lr_mag = []
    logger_val_weight_mse_mag = []
    logger_val_mse_mag = []
    logger_train_mse_0094 = []
    logger_train_weighted_mse_0094 = []
    logger_lr_0094 = []
    logger_val_weight_mse_0094 = []
    logger_val_mse_0094 = []

    start_time = time.time()
    iter_time = time.time()
    print(f'Start training Modal: {args.modal_list}')
    for epoch in range(args.epochs):
        pretrainModel_mag.train()
        pretrainModel_0094.train()
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            image_mag = data[:,0,:,:,:]
            image_0094 = data[:,1,:,:,:]
            image_mag = enhance_funciton(image_mag, args.enhance_list[0], args.enhance_list[1])
            image_0094 = enhance_funciton(image_0094, args.enhance_list[0], args.enhance_list[1])
            iteration_txt = f"Iteration {i} | Data time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            weights_mag,_ = get_weights(args.weights_type_list[0], image_mag)
            weights_0094,_ = get_weights(args.weights_type_list[1], image_0094)
            mse_with_weight_mag, mse_mag = pretrainModel_mag.calculate_loss(image_mag, weights_mag)
            mse_with_weight_0094, mse_0094 = pretrainModel_0094.calculate_loss(image_0094, weights_0094)
            iteration_txt += f" Forward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            optimizer_mag.zero_grad()
            optimizer_0094.zero_grad()
            mse_with_weight_mag.backward()
            mse_with_weight_0094.backward()      
            optimizer_mag.step()
            optimizer_0094.step()
            iteration_txt += f" Backward time: {(time.time()-epoch_time)/60:.2f} min |"
            epoch_time = time.time()

            logger_train_mse_mag.append(mse_mag.item())
            logger_train_weighted_mse_mag.append(mse_with_weight_mag.item())
            logger_lr_mag.append(scheduler_mag.get_last_lr()[0])
            logger_train_mse_0094.append(mse_0094.item())
            logger_train_weighted_mse_0094.append(mse_with_weight_0094.item())
            logger_lr_0094.append(scheduler_0094.get_last_lr()[0])
            iteration_txt += f" Iteration time: {(time.time()-iter_time)/60:.2f} min |"
            iter_time = time.time()
            print(iteration_txt)

        scheduler_mag.step()
        scheduler_0094.step()

        print(f'Epoch {epoch+1:>6}/{args.epochs:<6} | Train weight mse magnet {logger_train_weighted_mse_mag[-1]:<10.8f} | LR of magnet {logger_lr_mag[-1]:<10.8f} | Train weight mse 0094 {logger_train_weighted_mse_0094[-1]:<10.8f} | LR of 0094 {logger_lr_0094[-1]:<10.8f} | Cost {(time.time()-start_time)/60:.2f} min')

        if (epoch+1) % test_epoch == 0:
            with torch.no_grad():
                pretrainModel_mag.eval()
                pretrainModel_0094.eval()
                weights_mse_results_mag = []
                mse_results_mag = []   
                weights_mse_results_0094 = []
                mse_results_0094 = []

                for i, data in enumerate(val_loader):
                    data = data.to(args.device)
                    image_mag = data[:,0,:,:,:]
                    image_0094 = data[:,1,:,:,:]
                    image_mag = enhance_funciton(image_mag, args.enhance_list[0], args.enhance_list[1])
                    image_0094 = enhance_funciton(image_0094, args.enhance_list[0], args.enhance_list[1])
                    
                    weights_mag,_ = get_weights(args.weights_type_list[0], image_mag)
                    weights_0094,_ = get_weights(args.weights_type_list[1], image_0094)
                    weight_mse_mag, mse_mag = pretrainModel_mag.calculate_loss(image_mag, weights_mag)
                    weight_mse_0094, mse_0094 = pretrainModel_0094.calculate_loss(image_0094, weights_0094)
                    weights_mse_results_mag.append(weight_mse_mag.item())
                    mse_results_mag.append(mse_mag.item())
                    logger_lr_mag.append(scheduler_mag.get_last_lr()[0])
                    weights_mse_results_0094.append(weight_mse_0094.item())
                    mse_results_0094.append(mse_0094.item())
                    logger_lr_0094.append(scheduler_0094.get_last_lr()[0])

                logger_val_weight_mse_mag.append(np.mean(weights_mse_results_mag))
                logger_val_mse_mag.append(np.mean(mse_results_mag))
                logger_val_weight_mse_0094.append(np.mean(weights_mse_results_0094))
                logger_val_mse_0094.append(np.mean(mse_results_0094))
                
                result_txt = f'Epoch {epoch+1:>6}/{args.epochs:<6} | '
                result_txt += f'Magnet Train weight mse {logger_train_weighted_mse_mag[-1]:<10.8f} | '
                result_txt += f'Magnet Train mse {logger_train_mse_mag[-1]:<10.8f} | '
                result_txt += f'Magnet Val weight mse {logger_val_weight_mse_mag[-1]:<10.8f} | '
                result_txt += f'0094 Train weight mse {logger_val_weight_mse_0094[-1]:<10.8f} | '
                result_txt += f'0094 Train  mse {logger_val_mse_0094[-1]:<10.8f} | '
                result_txt += f'0094 Val weight mse {logger_val_weight_mse_0094[-1]:<10.8f} | '
                result_txt += f'Save time {(time.time()-start_time)/60:.2f} min'
                print(result_txt)

                with open(f'{logger_checkpoint_path_mag}logger_train_weighted_mse.pkl', 'wb') as f:
                    pickle.dump(logger_train_weighted_mse_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_train_mse.pkl', 'wb') as f:
                    pickle.dump(logger_train_mse_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_val_weight_mse.pkl', 'wb') as f:
                    pickle.dump(logger_val_weight_mse_mag, f)
                with open(f'{logger_checkpoint_path_mag}logger_val_mse.pkl', 'wb') as f:
                    pickle.dump(logger_val_mse_mag, f)
                with open(f'{logger_checkpoint_path_0094}logger_train_weighted_mse.pkl', 'wb') as f:
                    pickle.dump(logger_train_weighted_mse_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_train_mse.pkl', 'wb') as f:
                    pickle.dump(logger_train_mse_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_lr.pkl', 'wb') as f:
                    pickle.dump(logger_lr_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_val_weight_mse.pkl', 'wb') as f:
                    pickle.dump(logger_val_weight_mse_0094, f)
                with open(f'{logger_checkpoint_path_0094}logger_val_mse.pkl', 'wb') as f:
                    pickle.dump(logger_val_mse_0094, f)


        if (epoch+1) % save_epoch == 0:
            torch.save({'model': pretrainModel_mag.state_dict(), 'optimizer': optimizer_mag.state_dict(),
                        'scheduler': scheduler_mag.state_dict(), 'epoch': epoch},
                       f'{model_checkpoint_path_mag}epoch_{epoch+1}.pt')
            torch.save({'model': pretrainModel_0094.state_dict(), 'optimizer': optimizer_0094.state_dict(),
                        'scheduler': scheduler_0094.state_dict(), 'epoch': epoch},
                       f'{model_checkpoint_path_0094}epoch_{epoch+1}.pt')
            print(f'Model saved {(epoch+1)/args.epochs:.2%}, cost {(time.time()-start_time)/60:.2f} min')



def main():
    start_time = time.time()
    start_date = transfer_date_to_id(2010, 5, 1)
    end_date = transfer_date_to_id(2020, 6, 30)
    train_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                             start_date, end_date], modal_list=["magnet","0094"], load_imgs = True, enhance_list=[1024,0.5,90], batch_size=400, shuffle=True, num_workers=0)
    start_date = transfer_date_to_id(2020, 6, 30)
    end_date = transfer_date_to_id(2024, 6, 30)
    val_loader = Solardataloader_subset.get_loader_by_time(time_step=60*12,time_interval=[
                                                           start_date, end_date], modal_list=["magnet","0094"], load_imgs = True, enhance_list=[1024,0,0], batch_size=400, shuffle=True, num_workers=0)
    print(f"DataLoader time: {(time.time()-start_time)/60:.2f} min")

    args = parse_args()
    _ = args.config_path
    if args.config_path != 'None':
        args = load_args_from_json(args, args.config_path)
    args.config_path = _
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")

    train(args, train_loader, val_loader)

    

if __name__ == '__main__':
    main()  
    # for i in range(3):
    #     args = parse_args()
    #     args = load_args_from_json(args,config_path[i])
    #     print(args)
    #     modal = args.modal
    #     enhance_ls = args.enhance_list
    #     epochs = args.epochs
    #     test_freq = args.test_freq
    #     save_freq = args.save_freq
    #     dev = args.device

    #     train(train_loader, val_loader, epochs, test_freq, save_freq, modal, enhance_ls,dev)