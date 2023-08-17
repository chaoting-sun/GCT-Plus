#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=3
BENCHMARK=moses


MODEL_TYPE=vaetf
MODEL_NAME=${MODEL_TYPE}1_
EPOCH=37      

# MODEL_TYPE=scavaetf
# MODEL_NAME=${MODEL_TYPE}1
# EPOCH=15


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u   \
    inference.py                                                   \
    visualize-attention                                         \
        -visualize_attention                                    \
        -decode_algo greedy                                   \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -model_type ${MODEL_TYPE}                         \
        -model_name ${MODEL_NAME}                         \
        -epoch ${EPOCH}                                   \
        -n_jobs 8 \
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &
