#!/usr/bin/env bash

MLP_STACK=2
SIMILARITY=1.00
EPOCH=20
GPU_IDX=1

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     generate.py \
#         -variational \
#     testing \
#         -epoch ${EPOCH} \
#         -model_type transformer \
#         -decode_type decode \
#         -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${MLP_STACK} \
#         -demo

# mlp_transformer - mlp_decoder
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     generate.py \
#         -variational \
#         -model_type mlp_transformer \
#     testing \
#         -epoch ${EPOCH} \
#         -decode_type mlp_decode \
#         -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${MLP_STACK} \
#     >mlptf_generate_sim${SIMILARITY}_${MLP_STACK}.out 2>mlptf_generate_sim${SIMILARITY}_${MLP_STACK}.err &

# transformer - decoder
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    generate.py \
        -variational \
        -model_type transformer \
    testing \
        -decode_type decode \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/tf_sim1.00 \
    # >tf_generate.out 2>tf_generate.err &