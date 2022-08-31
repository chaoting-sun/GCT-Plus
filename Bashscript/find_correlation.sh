#!/usr/bin/env bash

SIMILARITY=1.00
MLP_STACK=1
EPOCH=3
GPU_IDX=1

export PYTHONPATH='/home/chaoting/tools/python-plot/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/scikit-learn/':$PYTHONPATH


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    find_correlation.py \
        -variational \
        -model_type mlp_encoder \
    testing \
        -epoch ${EPOCH} \
        -decode_type mlp_decode \
        -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK}_mse \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${MLP_STACK}_mse_epoch${EPOCH} \
    # >compute_correlation.out 2>compute_correlation.err &
