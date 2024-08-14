#!/bin/bash
NAME="mag_222324"

TAR_FILE="./tar/${NAME}.tar.gz"
CHECKPOINT_FILE="./log/{NAME}checkpoint.txt"
CHECKPOINT_INTERVAL=1000

if [ -f "$CHECKPOINT_FILE" ]; then
    LAST_CHECKPOINT=$(cat "$CHECKPOINT_FILE")
    echo "从 checkpoint $LAST_CHECKPOINT 继续解压..."
else
    LAST_CHECKPOINT=0
    echo "从头开始解压..."
fi

tar --skip-old-files \
    --checkpoint=$CHECKPOINT_INTERVAL \
    --checkpoint-action=exec='echo $TAR_CHECKPOINT >'"$CHECKPOINT_FILE" \
    -xzvf "$TAR_FILE" \
    -C ./try/data_test/hmi/magnet_pt > ./log/${NAME}.log 2>&1

if [ $? -eq 0 ]; then
    echo "解压完成，删除 checkpoint 文件。"
    rm -f "$CHECKPOINT_FILE"
else
    echo "解压中断，请稍后重新运行脚本继续解压。"
fi
# tar -xzvf ../tar/mag_131415.tar.gz -C ../data_tar/hmi/magnet_pt