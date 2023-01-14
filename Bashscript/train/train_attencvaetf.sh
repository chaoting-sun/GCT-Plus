#!/usr/bin/env bash

GPU_IDX=2

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
    train.py \
        -model_type attencvaetf \
        -tolerance 0.20  \
        -similarity 0.80 \
        -use_cvaetf_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/transformer/model_25.pt \
        -start_epoch 26   \
        -num_epoch 30    \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/attencvaetf_selftf \
    # >train_attencvaetf_selftf.out 2>&1 &