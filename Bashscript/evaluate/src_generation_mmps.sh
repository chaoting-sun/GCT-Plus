#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

# GPU_IDX=0
# MODEL_TYPE=cvaetf
# MODEL_NAME=cvaetf
# BENCHMARK=chembl_02

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                    \
#         -model_type ${MODEL_TYPE} \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset \
#     src-generation-mmps \
#         -src_generation_mmps \
#         -benchmark ${BENCHMARK} \
#         -property_list logP \
#         -decode_algo multinomial \
#         -model_name ${MODEL_NAME} \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -data_name test \
#         -n_steps 2 \
#         -epoch_list 20 \


GPU_IDX=1
MODEL_TYPE=cvaetf
MODEL_NAME=cvaetf
BENCHMARK=chembl_02

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py                                    \
        -model_type ${MODEL_TYPE} \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset \
    src-generation-mmps \
        -src_generation_mmps \
        -benchmark ${BENCHMARK} \
        -property_list logP \
        -decode_algo multinomial \
        -model_name ${MODEL_NAME} \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -data_name test \
        -n_steps 1 \
        -epoch_list 20 \
    >/dev/null 2>&1 &