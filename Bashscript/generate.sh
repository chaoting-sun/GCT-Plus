#!/usr/bin/env bash

MLP_STACK=1
SIMILARITY=0.90
EPOCH=1
GPU_IDX=2
LOSS_FCN=kld

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


# # mlp_transformer - mlp_decoder
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     generate.py \
#         -variational \
#         -model_type mlp_encoder \
#     testing \
#         -epoch ${EPOCH} \
#         -decode_type mlp_decode \
#         -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK}_kld \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${MLP_STACK}_epoch${EPOCH} \
#     >mlptf_generate_sim${SIMILARITY}_${MLP_STACK}_kld_epoch${EPOCH}.out 2>mlptf_generate_sim${SIMILARITY}_${MLP_STACK}_kld_epoch${EPOCH}.err &


# transformer - decoder
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     generate.py \
#         -variational \
#         -model_type transformer \
#     testing \
#         -decode_type decode \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/tf_sim1.00 \
    # >tf_generate.out 2>tf_generate.err &

# STD=1.0

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     generate.py \
#         -variational \
#         -model_type mlp_encoder \
#     testing \
#         -epoch ${EPOCH} \
#         -decode_type mlp_decode \
#         -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN} \
#         -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN}_epoch${EPOCH}_std${STD} \
#     >mlptf_generate_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN}_epoch${EPOCH}_${STD}.out 2>mlptf_generate_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN}_epoch${EPOCH}_${STD}.err &


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    generate.py \
        -variational \
        -model_type mlp_encoder \
    testing \
        -epoch ${EPOCH} \
        -decode_type mlp_decode \
        -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN}_epoch${EPOCH} \
    >mlptf_generate_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN}_epoch${EPOCH}.out 2>mlptf_generate_sim${SIMILARITY}_${MLP_STACK}_${LOSS_FCN}_epoch${EPOCH}.err &
