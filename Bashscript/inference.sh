#!/usr/bin/env bash 

MLP_STACK=1
SIMILARITY=1.00

CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 \
    python get_model.py \
        -variational \
    testing \
        -epoch 20 \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK} \