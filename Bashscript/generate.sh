#!/usr/bin/env bash 

MLP_STACK=3
SIMILARITY=0.70
EPOCH=20
GPU_IDX=1

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    generate.py \
        -variational \
    testing \
        -epoch ${EPOCH} \
        -model_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK} \
        -storage_path /fileserver-gamma/chaoting/ML/cvae-transformer/Inference/mlptf_sim${SIMILARITY}_${MLP_STACK} \
        -decode_type mlp_decode \
    >mlptf_generate_sim${SIMILARITY}_${MLP_STACK}.out 2>mlptf_generate_sim${SIMILARITY}_${MLP_STACK}.err &