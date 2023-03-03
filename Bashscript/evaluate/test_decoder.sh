#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=3
# MODEL_NAME=transformer_aug-s0.50-t0.10
MODEL_NAME=transformer

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py                                          \
    decoder-test                                          \
        -decoder_test                                     \
        -model_name ${MODEL_NAME}                         \
        -epoch_list 20                                    \
    >>test_decoder.out 2>&1 &