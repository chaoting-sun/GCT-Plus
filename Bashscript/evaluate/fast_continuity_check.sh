#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


# GPU_IDX=0

# MODEL_NAME=transformer

# MODEL_NAME=transformer_ep25_aug-s0.80-t0.10
# MODEL_NAME=transformer_ep25_aug-s0.70-t0.10
# MODEL_NAME=transformer_ep25_aug-s0.50-t0.10

# MODEL_NAME=transformer_ep25_aug-s0.60-t0.20

# MODEL_NAME=transformer_aug-s0.50-t0.10


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -model_type transformer             \
#         -model_name ${MODEL_NAME}           \
#         -epoch_list 15 16 17 18 19 20       \
#     >>ContiCheck_model-${MODEL_NAME}.out 2>&1 &

# GPU_IDX=2
# MODEL_NAME=transformer
# MODEL_NUM=3

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat \
#         -model_type transformer                     \
#         -model_name ${MODEL_NAME}${MODEL_NUM}       \
#         -epoch_list 19 20 21 22 23 24 25   \
#     >>ContiCheck_model-${MODEL_NAME}{MODEL_NUM}.out 2>&1 &


GPU_IDX=1
MODEL_NAME=transformer
DATA_NUM=140

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat \
    continuity-check                                            \
        -continuity_check                                       \
        -model_name ${MODEL_NAME}-${DATA_NUM}                   \
        -epoch_list 24 25                                       \
    >>ContiCheck_model-${MODEL_NAME}-${DATA_NUM}.out 2>&1 &