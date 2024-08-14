#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0  # Set the GPU device (e.g., 0 for the first GPU)

# Activate virtual environment if you're using one
# source /path/to/your/virtualenv/bin/activate

# Define directories and parameters
MAG_DIR="/mnt/nas/home/huxing/202407/nas/data/hmi/fits/2010"
H_DIR="/mnt/nas/home/zhouyuqing/downloads"
BATCH_SIZE=5
LEARNING_RATE=1e-2
EPOCHS=50
NUM_WORKERS=12
NUM_SHUFFLE=100000
EMBED_DIM=512
VISION_WIDTH=768
IMAGE_RESOLUTION_MAG=224
VISION_LAYERS_MAG=12
VISION_PATCH_SIZE_MAG=32
IMAGE_RESOLUTION_H=224
VISION_LAYERS_H=12
VISION_PATCH_SIZE_H=32


json_file1="./json/test/class_none/config_1.json"
cat <<EOF > $json_file1
{
    "mag_dir": "$MAG_DIR1",
    "h_dir": "$H_DIR1",
    "batch_size": $BATCH_SIZE1,
    "learning_rate": $LEARNING_RATE1,
    "epochs": $EPOCHS1,
    "num_workers": $NUM_WORKERS1,
    "num_shuffle": $NUM_SHUFFLE1,
    "embed_dim": $EMBED_DIM1,
    "vision_width": $VISION_WIDTH1,
    "image_resolution_mag": $IMAGE_RESOLUTION_MAG1,
    "vision_layers_mag": $VISION_LAYERS_MAG1,
    "vision_patch_size_mag": $VISION_PATCH_SIZE_MAG1,
    "image_resolution_H": $IMAGE_RESOLUTION_H1,
    "vision_layers_H": $VISION_LAYERS_H1,
    "vision_patch_size_H": $VISION_PATCH_SIZE_H1,
    "checkpoint_path": "./pt/classembedding/none1",
    "token_type": "class embedding"
}
EOF
# Run the training script

python -u Solarclip_test.py \
    --mag_dir $MAG_DIR \
    --h_dir $H_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --num_workers $NUM_WORKERS \
    --num_shuffle $NUM_SHUFFLE \
    --embed_dim $EMBED_DIM \
    --vision_width $VISION_WIDTH \
    --image_resolution_mag $IMAGE_RESOLUTION_MAG \
    --vision_layers_mag $VISION_LAYERS_MAG \
    --vision_patch_size_mag $VISION_PATCH_SIZE_MAG \
    --image_resolution_H $IMAGE_RESOLUTION_H \
    --vision_layers_H $VISION_LAYERS_H \
    --vision_patch_size_H $VISION_PATCH_SIZE_H \
    --checkpoint_path "./pt/classembedding/none" \
    --token_type "class embedding"\
    2>&1 | tee test_clip_class_none.out

python -u Solarclip_test.py \
    --mag_dir $MAG_DIR \
    --h_dir $H_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --num_workers $NUM_WORKERS \
    --num_shuffle $NUM_SHUFFLE \
    --embed_dim $EMBED_DIM \
    --vision_width $VISION_WIDTH \
    --image_resolution_mag $IMAGE_RESOLUTION_MAG \
    --vision_layers_mag $VISION_LAYERS_MAG \
    --vision_patch_size_mag $VISION_PATCH_SIZE_MAG \
    --image_resolution_H $IMAGE_RESOLUTION_H \
    --vision_layers_H $VISION_LAYERS_H \
    --vision_patch_size_H $VISION_PATCH_SIZE_H \
    --checkpoint_path "./pt/allembedding/none" \
    --modal_list magnet 0094 \
    --token_type "all embedding"\
    2>&1 | tee test_clip_all_none.out



# python -u Solarclip_train.py \
#     --mag_dir $MAG_DIR \
#     --h_dir $H_DIR \
#     --batch_size $BATCH_SIZE \
#     --learning_rate $LEARNING_RATE \
#     --epochs $EPOCHS \
#     --num_workers $NUM_WORKERS \
#     --num_shuffle $NUM_SHUFFLE \
#     --embed_dim $EMBED_DIM \
#     --vision_width $VISION_WIDTH \
#     --image_resolution_mag $IMAGE_RESOLUTION_MAG \
#     --vision_layers_mag $VISION_LAYERS_MAG \
#     --vision_patch_size_mag $VISION_PATCH_SIZE_MAG \
#     --image_resolution_H $IMAGE_RESOLUTION_H \
#     --vision_layers_H $VISION_LAYERS_H \
#     --vision_patch_size_H $VISION_PATCH_SIZE_H \
#     --output_path "./pt/classembdding/none" \
#     --token_type "class embedding"\
#     2>&1 | tee train_clip_class_log1p



# python -u Solarclip_train.py \
#     --mag_dir $MAG_DIR \
#     --h_dir $H_DIR \
#     --batch_size $BATCH_SIZE \
#     --learning_rate $LEARNING_RATE \
#     --epochs $EPOCHS \
#     --num_workers $NUM_WORKERS \
#     --num_shuffle $NUM_SHUFFLE \
#     --embed_dim $EMBED_DIM \
#     --vision_width $VISION_WIDTH \
#     --image_resolution_mag $IMAGE_RESOLUTION_MAG \
#     --vision_layers_mag $VISION_LAYERS_MAG \
#     --vision_patch_size_mag $VISION_PATCH_SIZE_MAG \
#     --image_resolution_H $IMAGE_RESOLUTION_H \
#     --vision_layers_H $VISION_LAYERS_H \
#     --vision_patch_size_H $VISION_PATCH_SIZE_H \
#     --output_path "./pt/allembdding/log1p" \
#     --token_type "all embedding" \
#     2>&1 | tee train_clip_all_log1p.out






