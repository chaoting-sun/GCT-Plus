#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=0
BENCHMARK=moses


MODEL_TYPE=scacvaetfv3
MODEL_NAME=${MODEL_TYPE}-warmup15000
EPOCH=8


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
        -use_cond2lat \
    scaffold-sampling                                  \
        -scaffold_sampling                         \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -decode_algo multinomial                                         \
        -model_type ${MODEL_TYPE}                                        \
        -model_name ${MODEL_NAME}                                        \
        -property_list logP tPSA QED                                     \
        -epoch ${EPOCH}                        \
    # >>scaffold_sampling-${MODEL_NAME}.out 2>&1 &


    # -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
