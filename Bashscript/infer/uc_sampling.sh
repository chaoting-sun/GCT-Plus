#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=2
EPOCH=37
MODEL=vaetf1

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u                                                           \
    inference.py                                                                                                           \
    uc-sampling                                                                                                            \
        -decode_algo multinomial                                                                                           \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/                                                   \
        -model_type vaetf                                                                                                  \
        -model_name model_${EPOCH}.pt                                                                                      \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL}/                    \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/uc-sampling/${MODEL}-${EPOCH}/ \
        -n_jobs 8                                                                                                          \
        -n_samples 30000
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &
