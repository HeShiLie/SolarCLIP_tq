import torch
import torch.nn.functional as F
from einops import rearrange
from Model.SolarCLIP import SolarCLIP_MODEL

import matplotlib.pyplot as plt
import numpy as np

from Model.get_weights import get_weights
def plot_matrix_with_images(cor_matrix, inner_cor_matrix,row_images, col_images, save_path=None,inner_loss_rate=0):
    
    border_width = 2
    border_color = 'black'

    num_rows, num_cols = cor_matrix.shape

    vmin_magnet = np.min(row_images)
    vmax_magnet = np.max(row_images)
    vmax_magnet = np.max([np.abs(vmin_magnet), np.abs(vmax_magnet)])/2
    vmin_magnet = -vmax_magnet

    vmin_0094 = np.min(col_images)
    vmax_0094 = np.max(col_images)

    # cor_matrix = np.exp(cor_matrix*10)
    # cor_matrix = cor_matrix / np.max(cor_matrix)
    vmin_cor_matrix = -np.max(np.abs(cor_matrix))
    vmax_cor_matrix = -vmin_cor_matrix

    cor_matrix_row_max_indices = np.argmax(cor_matrix, axis=1)
    cor_matrix_col_max_indices = np.argmax(cor_matrix, axis=0)

    fig, ax = plt.subplots(num_rows+1, num_cols+1, figsize=(2*(num_cols+1), 2*(num_rows+1)))
    # Plot the matrix
    for i in range(num_rows):
        for j in range(num_cols):
            if inner_loss_rate == 0:
                if j == cor_matrix_row_max_indices[i] :  #优先展示行最大
                    ax[i+1, j+1].text(0.5, 0.5, f'{cor_matrix[i, j]:.4f}', transform=ax[i+1, j+1].transAxes,
                            ha='center', va='center', fontsize=20, color='red',fontweight='bold')
                    full_color = np.ones([2*(num_cols+1), 2*(num_rows+1)])*cor_matrix[i, j]
                    ax[i+1, j+1].imshow(full_color, cmap='RdBu',vmin=vmin_cor_matrix, vmax=vmax_cor_matrix)
                    ax[i+1, j+1].set_xticks([])
                    ax[i+1, j+1].set_yticks([])
                elif i == cor_matrix_col_max_indices[j]:
                    ax[i+1, j+1].text(0.5, 0.5, f'{cor_matrix[i, j]:.4f}', transform=ax[i+1, j+1].transAxes,
                            ha='center', va='center', fontsize=20, color='black',fontweight='bold')
                    full_color = np.ones([2*(num_cols+1), 2*(num_rows+1)])*cor_matrix[i, j]
                    ax[i+1, j+1].imshow(full_color, cmap='RdBu',vmin=-0.7, vmax=0.7)
                    ax[i+1, j+1].set_xticks([])
                    ax[i+1, j+1].set_yticks([])
                    pass
                else:
                    ax[i+1, j+1].text(0.5, 0.5, f'{cor_matrix[i, j]:.4f}', transform=ax[i+1, j+1].transAxes,
                                ha='center', va='center', fontsize=12, color='black')           
                    full_color = np.ones([2*(num_cols+1), 2*(num_rows+1)])*cor_matrix[i, j]
                    ax[i+1, j+1].imshow(full_color, cmap='RdBu',vmin=-0.7, vmax=0.7)
                    ax[i+1, j+1].set_xticks([])
                    ax[i+1, j+1].set_yticks([])
            else:
                
                pass



    # Plot magnet images
    for i in range(num_rows):
        ax[i+1, 0].imshow(row_images[i][0], cmap='RdBu_r',vmin=vmin_magnet, vmax=vmax_magnet)
        ax[i+1, 0].set_xticks([])
        ax[i+1, 0].set_yticks([])

    # Plot 0094 images
    for j in range(num_cols):
        ax[0, j+1].imshow(col_images[j][0], cmap='Reds',vmin=vmin_0094, vmax=vmax_0094)
        ax[0, j+1].set_xticks([])
        ax[0, j+1].set_yticks([])

    # Turn off the top-left empty subplot
    ax[0, 0].axis('off')

    # set border line width and color
    for i in range(num_rows+1):
        for j in range(num_cols+1):
            for spine in ax[i, j].spines.values():
                spine.set_linewidth(border_width)
                spine.set_edgecolor(border_color)
    # set border spacing
    plt.subplots_adjust(wspace=0, hspace=0)
    

    plt.show()

    if save_path:
        plt.savefig(save_path)
    

def calculate_loss(model,batch, inner_loss_rate = 0, criterion = torch.nn.functional.cross_entropy):
    mag_image = batch[:,0,:,:,:] # [batch, channel, height, width]
    h_image = batch[:,1,:,:,:]

    logits_per_mag, logits_per_h, inner_cor_matrix = model(mag_image, h_image)
    ground_truth = torch.arange(len(mag_image), dtype=torch.long, device=mag_image.device)
    
    loss_img = criterion(logits_per_mag, ground_truth)
    loss_h = criterion(logits_per_h, ground_truth)
    loss = (loss_img + loss_h) / 2
    acc = (torch.argmax(logits_per_mag, dim=1) == ground_truth).float().mean().item()

    assert inner_loss_rate >=0
    if inner_loss_rate > 0:
        ground_truth = torch.arange(inner_cor_matrix.shape[-1], dtype=torch.long, device=inner_cor_matrix.device)#[L]
        loss_inner = criterion(inner_cor_matrix,ground_truth)/2
        loss_inner = loss_inner + criterion(inner_cor_matrix.t(),ground_truth)/2
    else:
        loss_inner = torch.tensor(0, dtype=torch.float32, device=inner_cor_matrix.device)

    return loss, loss_inner, acc, logits_per_mag, inner_cor_matrix
    # return loss, acc, logits_per_mag

def calculate_loss_reconstruction(decode_modal,decoder, batch, encoderModel, criterion = torch.nn.functional.mse_loss,
                                  weights_type = '3sigma-discrete'):
    mag_image = batch[:,0,:,:,:] # [batch, channel, height, width]  batch:[batch, modal, channel, height, width]
    h_image = batch[:,1,:,:,:]
    mag_weights, _ = get_weights(weights_type, mag_image)
    h_weights, _ = get_weights(weights_type, h_image )
    mag_feature, h_feature = encoderModel.encode_mag(mag_image), encoderModel.encode_H(h_image)
    mag_feature, h_feature = mag_feature[:,1:,:], h_feature[:,1:,:]  ##all embedding
    if decode_modal == 'magnet':
        mag_recon = decoder(mag_feature)
        loss = criterion(mag_recon, mag_image)
        mae = mag_weights*F.l1_loss(mag_recon, mag_image).float().item()  
    elif decode_modal == '0094':
        h_recon = decoder(h_feature)
        loss = criterion(h_recon, h_image)
        mae = h_weights*F.l1_loss(h_recon, h_image).float().item()
    else:
        raise ValueError('decode_modal should be magnet or 0094')
        
    return loss, mae

def calculate_loss_pretrain(modal,encoder,decoder, batch, criterion = torch.nn.functional.mse_loss, weights_type = '3sigma-discrete'):
    # [batch, channel, height, width]  batch:[batch, modal, channel, height, width]
    if modal == 'magnet':
        image = batch[:,0,:,:,:]
    elif modal == '0094':
        image = batch[:,1,:,:,:]
    weights, _ = get_weights(weights_type, image)
    feature = encoder(image)
    feature = feature[:,1:,:]  ##all embedding
    recon = decoder(feature)
    loss = weights*criterion(recon, image) 
    return loss




if __name__ == '__main__':
    model = SolarCLIP_MODEL(
        image_resolution_mag = 224,
        vision_layers_mag = 12,
        vision_patch_size_mag = 32,
        image_resolution_H = 224,
        vision_layers_H = 12,
        vision_patch_size_H = 32,
        transformer_token_type = 'class embedding'
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = torch.randn(1, 2, 224, 224)
    # test(model=model, batch=batch,device=device)

    a = torch.randn(32,32,5,5)
    torch.einsum('ijij->ij', a).shape