#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=3
BENCHMARK=moses


MODEL_TYPE=cvaetf
MODEL_NAME=${MODEL_TYPE}3
EPOCH=8


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u   \
    inference.py                                                   \
        -use_cond2lat \
    prop-sampling                                         \
        -property_list logP tPSA QED \
        -prop_sampling                                    \
        -decode_algo multinomial                                   \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -model_type ${MODEL_TYPE}                         \
        -model_name ${MODEL_NAME}                         \
        -epoch ${EPOCH}                                   \
        -n_jobs 16 \
    >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &