#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=1

MODEL_TYPE=scacvaetfv1
BENCHMARK=moses
MODEL_NAME=${MODEL_TYPE}-s1.00
EPOCH=12

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                          \
    encoder-test                                          \
        -encoder_test                                     \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -model_type ${MODEL_TYPE}                         \
        -model_name ${MODEL_NAME}                         \
        -epoch ${EPOCH}                                   \