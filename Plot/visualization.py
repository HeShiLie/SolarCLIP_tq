import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import ipywidgets as widgets
import io
import matplotlib.pyplot as plt

def transform(img, img_size):
     img = transforms.Resize(img_size)(img)
     img = transforms.ToTensor()(img)
     return img
 
def visualize_predict(model, img,  patch_size, device):
    # img_pre = transform(img, img_size)
    attention = visualize_attention(model, img, patch_size, device)
    plot_attention(img, attention)
 
def visualize_attention(model, img, patch_size, device):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0) # (1, 1 , 1024, 1024)

    w_featmap = img.shape[-2] // patch_size  # 1024/64 = 16
    print(w_featmap)
    h_featmap = img.shape[-1] // patch_size
    print(h_featmap)

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions

def plot_attention(img, attention):
    n_heads = attention.shape[0]


    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.show()
 
