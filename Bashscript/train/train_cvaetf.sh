#!/usr/bin/env bash 

GPU=1
BENCHMARK=moses


MODELTYPE=cvaetf
MODEL_NAME=${MODELTYPE}-QED
MODEL_FOLDER=/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME}

CUDA_VISIBLE_DEVICES=${GPU} CUDA_LAUNCH_BLOCKING=1 torchrun --master_port 29501 \
    Train.py                          \
        -use_cond2lat                 \
        -benchmark ${BENCHMARK}       \
        -start_epoch 1                \
        -num_epoch 50                 \
        -batch_size 128               \
        -model_type ${MODELTYPE}      \
        -property_list QED            \
        -model_folder ${MODEL_FOLDER} \
    >train-${MODEL_NAME}-${GPU}.out 2>&1 &


# MODELTYPE=scacvaetfv1
# MODEL_NAME=${MODELTYPE}-s${SIMILARITY}
# MODEL_FOLDER=/fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME}


# CUDA_VISIBLE_DEVICES=1,2 CUDA_LAUNCH_BLOCKING=1 nohup torchrun --master_port 29501 \
#     Train.py                          \
#         -benchmark ${BENCHMARK}       \
#         -use_cond2lat                 \
#         -start_epoch 1                \
#         -num_epoch 50                 \
#         -batch_size 64                \
#         -model_type ${MODELTYPE}      \
#         -property_list logP tPSA QED  \
#         -model_folder ${MODEL_FOLDER} \
        # -use_scaffold                 \
    # >train-${MODEL_NAME}.out 2>&1 &