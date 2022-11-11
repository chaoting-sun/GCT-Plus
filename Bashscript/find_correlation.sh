#!/usr/bin/env bash

SIMILARITY=1.00
MLP_STACK=1
EPOCH=1
GPU_IDX=1
MODEL=mlptf
LOSS_FCN=kld

export PYTHONPATH='/home/chaoting/tools/python-plot/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/scikit-learn/':$PYTHONPATH

# multi-faceted analysis
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     find_correlation.py \
#         -variational \
#         -model_type mlp_encoder \
#     testing \
#         -epoch ${EPOCH} \
#         -encode_type encode_sample_mlp_sample \
#         -decode_type mlp_decode \
#         -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/correlation_analysis/${MODEL}_sim${SIMILARITY}_${LOSS_FCN}_e${EPOCH} \
    # >compute_correlation.out 2>compute_correlation.err &

# # analysis on strings with the same length
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     find_correlation.py \
#         -variational \
#         -model_type mlp_encoder \
#     testing \
#         -epoch ${EPOCH} \
#         -encode_type encode_sample_mlp_sample \
#         -decode_type mlp_decode \
#         -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/correlation_analysis/${MODEL}_sim${SIMILARITY}_${LOSS_FCN}_e${EPOCH}_same_len \
#     # >compute_correlation.out 2>compute_correlation.err &

# multi-faceted analysis
CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    find_correlation.py \
        -variational \
        -model_type mlp_encoder \
    testing \
        -epoch ${EPOCH} \
        -encode_type encode_sample_mlp_sample \
        -decode_type mlp_decode \
        -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${LOSS_FCN} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/correlation_analysis/${MODEL}_sim${SIMILARITY}_${LOSS_FCN}_epoch${EPOCH} \
    # >compute_correlation.out 2>compute_correlation.err &