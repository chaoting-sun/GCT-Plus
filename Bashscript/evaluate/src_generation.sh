#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=0
MODEL_NAME=transformer

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
    src-generation                                  \
        -src_generation                             \
        -model_name ${MODEL_NAME}                   \
        -epoch_list 20                              \
        -src_smiles 'CNC(=O)c1cccc(NCC(=O)Nc2cccc(C(=O)NC)c2)c1' \
        -trg_props 1 80 0.8                         \
    # >>src-generation_model-${MODEL_NAME}.out 2>&1 &
