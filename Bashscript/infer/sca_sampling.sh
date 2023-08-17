#!/usr/bin/env bash


##### Our settings


# GPU_IDX=1
# EPOCH=15
# MODEL=scavaetf1
# SCAFFOLD_SOURCE=train


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u         \
#     inference.py                                                         \
#         -use_cond2lat                                                    \
#         -use_scaffold                                                    \
#     sca-sampling                                                         \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/        \
#         -decode_algo multinomial                                         \
#         -model_type scavaetf                                             \
#         -model_name model_${EPOCH}.pt                                    \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL}/ \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling/${MODEL}-${EPOCH}/${SCAFFOLD_SOURCE}/ \
#         -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling/ \
#         -n_samples 10000                                                 \
#         -scaffold_source ${SCAFFOLD_SOURCE}                              \
    # >>sca_sampling-${MODEL_NAME}-${GPU_IDX}.out 2>&1 & 


##### MolGPT settings


GPU_IDX=3
EPOCH=16
MODEL=scavaetf3


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u         \
    inference.py                                                         \
        -use_cond2lat                                                    \
        -use_scaffold \
    sca-sampling                                                         \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/        \
        -decode_algo multinomial                                         \
        -model_type scavaetf                                             \
        -model_name model_${EPOCH}.pt                                    \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL}/ \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling/${MODEL}-${EPOCH}/molgpt/ \
        -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling/ \
        -n_samples 100                                                   \
        -scaffold_source test_scaffolds                                  \
    # >>sca_sampling-${MODEL_NAME}-${GPU_IDX}.out 2>&1 & 
