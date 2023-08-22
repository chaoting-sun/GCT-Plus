#!/usr/bin/env bash


##### Our settings


MODEL_NAME=scavaetf1
SCAFFOLD_SOURCE=train # or test_scaffolds


CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
        -use_cond2lat \
        -use_scaffold \
    sca-sampling \
        -model_type scavaetf \
        -model_name ${MODEL_NAME}.pt \
        -model_folder ./Weights/scavaetf \
        -save_folder ./Data/inference/sca-sampling/${MODEL_NAME}/${SCAFFOLD_SOURCE} \
        -scaffold_source ${SCAFFOLD_SOURCE} \
        -decode_algo multinomial \
        -n_samples 10000 \
    # >>sca_sampling.out 2>&1 &


##### MolGPT settings


# MODEL_NAME=scavaetf1
# SCAFFOLD_SOURCE=test_scaffolds


# CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     sca-sampling \
#         -model_type scavaetf \
#         -model_name scavaetf1.pt \
#         -model_folder ./Weights/scavaetf/ \
#         -save_folder ./Data/inference/sca-sampling/${MODEL_NAME}/molgpt \
#         -scaffold_source ${SCAFFOLD_SOURCE} \
#         -decode_algo multinomial \
#         -n_samples 100 \
#     >>sca_sampling.out 2>&1 & 
