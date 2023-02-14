#!/usr/bin/env bash 

GPU_IDX=3

TOLERANCE=0.00
SIMILARITY=0.00

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -model_type cvaetfcut \
        -tolerance ${TOLERANCE}   \
        -similarity ${SIMILARITY} \
        -start_epoch 1   \
        -num_epoch 30    \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/cvaetfcut_aug-s${SIMILARITY}-t${TOLERANCE} \
    >train_cvaetfcut_s${SIMILARITY}-t${TOLERANCE}.out 2>&1 &


