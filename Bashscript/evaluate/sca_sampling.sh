#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=2

MODEL_TYPE=scavaetf
MODEL_NAME=${MODEL_TYPE}1-warmup15000
# MODEL_NAME=${MODEL_TYPE}3-beta0.015-warmup15000
EPOCH=15


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u   \
    inference.py                                                         \
        -use_cond2lat                                                    \
    sca-sampling                                                         \
        -sca_sampling                                                    \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/        \
        -decode_algo multinomial                                         \
        -model_type ${MODEL_TYPE}                                        \
        -model_name ${MODEL_NAME}                                        \
        -epoch ${EPOCH}                                                  \
        -n_samples 100                                                   \
        -sample_from test_scaffolds                                      \
    # >>sca_sampling-${MODEL_NAME}-${GPU_IDX}.out 2>&1 & 


    # -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
