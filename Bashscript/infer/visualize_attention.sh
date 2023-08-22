#!/usr/bin/env bash


##### vaetf


MODEL_NAME=vaetf1


CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -get_attn \
    visualize-attention \
        -model_type vaetf \
        -model_name ${MODEL_NAME}.pt \
        -model_folder ./Weights/vaetf \
        -save_folder ./Data/inference/visualize-attention/${MODEL_NAME} \
        -smiles "CC(Cc1ccc(c(c1)OC)O)N" \
        -decode_algo greedy \
    # >>visualize-attention_${MODEL_NAME}.out 2>&1 &


##### scavaetf


# MODEL_NAME=scavaetf1


# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -use_scaffold \
#         -get_attn \
#     visualize-attention \
#         -model_type scavaetf \
#         -model_name ${MODEL_NAME}.pt \
#         -model_folder ./Weights/scavaetf \
#         -save_folder ./Data/inference/visualize-attention/${MODEL_NAME} \
#         -smiles "CC(Cc1ccc(c(c1)OC)O)N" \
#         -decode_algo greedy \
#     >>visualize-attention_${MODEL_NAME}.out 2>&1 &
