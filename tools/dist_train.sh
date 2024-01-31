#!/usr/bin/env bash

CONFIG='configs/shuffle_trans/mask_rcnn_shuffle_trans_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py'
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config $CONFIG --launcher pytorch ${@:3}
