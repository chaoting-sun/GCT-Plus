#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


GPU_IDX=3
BENCHMARK=moses
MODEL=cvaetf2
EPOCH=15


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u   \
    inference.py                                                   \
        -use_cond2lat \
    p-sampling                                         \
        -property_list logP tPSA QED \
        -decode_algo multinomial                                   \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
        -model_type cvaetf                                \
        -model_name model_${EPOCH}.pt               \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/p-sampling/${MODEL}-${EPOCH} \
        -n_jobs 8 \
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 & \


### use molgct model

# GPU_IDX=1
# BENCHMARK=moses
# MODEL=molgct
# EPOCH=15


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u   \
#     inference.py                                                   \
#         -use_cond2lat \
#     p-sampling                                         \
#         -property_list logP tPSA QED \
#         -decode_algo multinomial                                   \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
#         -model_type cvaetf                                \
#         -model_name molgct.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/molGCT/ \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling/molgct \
#         -n_jobs 8 \