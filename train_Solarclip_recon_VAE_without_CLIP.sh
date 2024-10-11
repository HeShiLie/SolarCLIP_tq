#!/bin/bash
# torchrun --nnodes=1 --nproc_per_node=3 --master_addr="10.200.48.108" --master_port=9000 Train_Solarclip_recon_DiTlike_VAEpretrain.py \
# 2>&1 | tee ./log/recon/DiTlike/VAEpretrain/trainAdamW1.out

# mkdir -p ./log/recon/VAE_without_CLIP/hidden_64/magnet-magnet
# mkdir -p ./log/recon/VAE_without_CLIP/hidden_64_block3/magnet-magnet
# mkdir -p ./log/recon/VAE_without_CLIP/hidden_96_block3/magnet-magnet
# mkdir -p ./log/recon/VAE_without_CLIP/hidden_128/magnet-magnet
# mkdir -p ./log/recon/VAE_without_CLIP/hidden_128_block3/magnet-magnet
# mkdir -p ./log/recon/VAE_without_CLIP/hidden_160/magnet-magnet
mkdir -p "./log/recon/VAE_without_CLIP/layers_3_kernels_[7,7,3]_strides_[4,4,2]/magnet-magnet"


# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_addr="10.200.48.105" --master_port=9012 Train_Solarclip_recon_VAE_without_CLIP.py \
# --hidden_dim 64 \
# --lambda_ratio 0 \
# --checkpoint_path /mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/VAE_wituout_CLIP/hidden_64 \
# 2>&1 | tee ./log/recon/VAE_without_CLIP/hidden_64/magnet-magnet/trainAdamW1_lambda_1e-1_cosine_lr_4e-4.out &

# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 --master_addr="10.200.48.105" --master_port=9990 Train_Solarclip_recon_VAE_without_CLIP.py \
# --hidden_dim 64 \
# --layers 2 \
# --kernel_sizes 7 7 \
# --strides 4 4 \
# --lambda_ratio 1e-1 \
# --learning_rate 1e-4 \
# --checkpoint_path /mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/VAE_wituout_CLIP_part2 \
# 2>&1 | tee "./log/recon/VAE_without_CLIP/layers_2_kernels_[7,7]_strides_[4,4]/magnet-magnet/hidden_64_trainAdamW1_lambda_1e-1_cosine_lr_1e-4.out" &

CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1 --master_addr="10.200.48.105" --master_port=9991 Train_Solarclip_recon_VAE_without_CLIP.py \
--hidden_dim 64 \
--layers 2 \
--kernel_sizes 7 7 \
--strides 4 4 \
--lambda_ratio 1e-1 \
--learning_rate 1e-4 \
--not_use_weight \
--checkpoint_path /mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/VAE_wituout_CLIP_part2 \
2>&1 | tee "./log/recon/VAE_without_CLIP/layers_2_kernels_[7,7]_strides_[4,4]/magnet-magnet/weight_false_hidden_64_trainAdamW1_lambda_1e-1_cosine_lr_1e-4.out" &

# CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --nproc_per_node=1 --master_addr="10.200.48.105" --master_port=9992 Train_Solarclip_recon_VAE_without_CLIP.py \
# --hidden_dim 128 \
# --layers 2 \
# --kernel_sizes 7 7 \
# --strides 4 4 \
# --lambda_ratio 1e-1 \
# --learning_rate 1e-4 \
# --checkpoint_path /mnt/nas/home/huxing/202407/ctf/SolarCLIP_tq/checkpoints/recon/VAE_wituout_CLIP_part2 \
# 2>&1 | tee "./log/recon/VAE_without_CLIP/layers_2_kernels_[7,7]_strides_[4,4]/magnet-magnet/hidden_128_trainAdamW1_lambda_1e-1_cosine_lr_1e-4.out" &


