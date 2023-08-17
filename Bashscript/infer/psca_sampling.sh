#!/usr/bin/env bash


##### Our settings


##### 1. train


MODEL=pscavaetf1
GPU_IDX=2
EPOCH=20
SCAFFOLD_SOURCE=train


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
        -use_cond2lat                               \
        -use_scaffold \
    psca-sampling                                   \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
        -decode_algo multinomial                                         \
        -model_type pscavaetf                                        \
        -model_name model_${EPOCH}.pt               \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL}/ \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/${MODEL}-${EPOCH}/${SCAFFOLD_SOURCE}/ \
        -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/ \
        -property_list logP tPSA QED \
        -n_samples 10000 \
        -scaffold_source ${SCAFFOLD_SOURCE} \
    # >>psca_sampling-${MODEL}-${GPU_IDX}.out 2>&1 &


##### 2. test_scaffolds


# MODEL=pscavaetf1
# GPU_IDX=2
# EPOCH=20
# SCAFFOLD_SOURCE=test_scaffolds


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     psca-sampling \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
#         -decode_algo multinomial \
#         -model_type pscavaetf \
#         -model_name model_${EPOCH}.pt \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/${MODEL}-${EPOCH}/${SCAFFOLD_SOURCE}/ \
#         -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/ \
#         -property_list logP tPSA QED \
#         -n_samples 10000 \
#         -scaffold_source ${SCAFFOLD_SOURCE} \
#     >>psca_sampling-${MODEL}-${GPU_IDX}.out 2>&1 &


##### MolGPT settings


# GPU_IDX=0
# MODEL=pscavaetf1_molgpt
# SCAFFOLD_SOURCE=molgpt


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     psca-sampling \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
#         -decode_algo multinomial \
#         -model_type pscavaetf \
#         -model_name model_${EPOCH}.pt \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL}/ \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/${MODEL}-${EPOCH}/${SCAFFOLD_SOURCE}/ \
#         -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/psca-sampling/ \
#         -property_list logP tPSA SAS \
#         -n_samples 10000 \
#         -scaffold_source ${SCAFFOLD_SOURCE} \
#     # >>psca_sampling-${MODEL}-${GPU_IDX}.out 2>&1 &
