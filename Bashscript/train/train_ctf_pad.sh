#!/usr/bin/env bash 

GPU_IDX=0

TOLERANCE=0.00
SIMILARITY=1.00

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
    train.py \
        -model_type ctf \
        -tolerance ${TOLERANCE}   \
        -similarity ${SIMILARITY} \
        -start_epoch 1   \
        -num_epoch 10    \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/ctf_pad \
        -pad_to_same_len \
    >train_ctf_pad.out 2>&1 &