#!/usr/bin/env bash 

GPU_IDX=1

TOLERANCE=0.00
SIMILARITY=1.00
THRESHOLD=0.001

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -model_type cvaetfcut \
        -tolerance ${TOLERANCE}   \
        -similarity ${SIMILARITY} \
        -start_epoch 1   \
        -num_epoch 30    \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/cvaetfcut_aug-s${SIMILARITY}-t${TOLERANCE}_${THRESHOLD} \
    >train_cvaetfcut_s${SIMILARITY}-t${TOLERANCE}.out 2>&1 &


