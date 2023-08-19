#!/usr/bin/env bash


##### vae


MODEL_NAME=vaetf1
PAIR_SOURCE=test_scaffolds # test_scaffolds-same_scaffold


CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py \
    mol-interpolation \
        -model_type vaetf \
        -model_name ${MODEL_NAME}.pt \
        -model_folder ./Weights/vaetf \
        -save_folder ./Data/inference/mol-interpolation/${MODEL_NAME}/${PAIR_SOURCE} \
        -pair_folder ./Data/molecular-pair \
        -pair_source ${PAIR_SOURCE} \
        -decode_algo greedy \
   >>mol-interpolation.out 2>&1 & \


##### pvaetf


# MODEL_NAME=pvaetf1
# PAIR_SOURCE=test_scaffolds # test_scaffolds-same_scaffold


# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -use_cond2lat \
#     mol-interpolation \
#         -property_list logP tPSA QED \
#         -model_type pvaetf \
#         -model_name ${MODEL_NAME}.pt \
#         -model_folder ./Weights/pvaetf \
#         -save_folder ./Data/inference/mol-interpolation/${MODEL_NAME}/${PAIR_SOURCE} \
#         -pair_folder ./Data/molecular-pair \
#         -pair_source ${PAIR_SOURCE} \
#         -decode_algo greedy \
#     >>mol-interpolation.out 2>&1 & \


##### scavaetf


# MODEL_NAME=scavaetf1
# PAIR_SOURCE=test_scaffolds # test_scaffolds-same_scaffold


# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     mol-interpolation \
#         -model_type scavaetf \
#         -model_name ${MODEL_NAME}.pt \
#         -model_folder ./Weights/scavaetf \
#         -save_folder ./Data/inference/mol-interpolation/${MODEL_NAME}/${PAIR_SOURCE} \
#         -pair_folder ./Data/molecular-pair \
#         -pair_source ${PAIR_SOURCE} \
#         -decode_algo greedy \
#     >>mol-interpolation.out 2>&1 & \


##### pscavaetf


# MODEL_NAME=pscavaetf1
# PAIR_SOURCE=test_scaffolds # test_scaffolds-same_scaffold


# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     mol-interpolation \
#         -property_list logP tPSA QED \
#         -model_type pscavaetf \
#         -model_name ${MODEL_NAME}.pt \
#         -model_folder ./Weights/pscavaetf \
#         -save_folder ./Data/inference/mol-interpolation/${MODEL_NAME}/${PAIR_SOURCE} \
#         -pair_folder ./Data/molecular-pair \
#         -pair_source ${PAIR_SOURCE} \
#         -decode_algo greedy \
#     >>mol-interpolation.out 2>&1 & \


##### ptf


# MODEL_NAME=ptf1
# PAIR_SOURCE=test_scaffolds # test_scaffolds-same_scaffold


# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py \
#         -use_cond2lat \
#         -use_scaffold \
#     mol-interpolation \
#         -property_list logP tPSA QED \
#         -model_type ptf \
#         -model_name ${MODEL_NAME}.pt \
#         -model_folder ./Weights/pscavaetf \
#         -save_folder ./Data/inference/mol-interpolation/${MODEL_NAME}/${PAIR_SOURCE} \
#         -pair_folder ./Data/molecular-pair \
#         -pair_source ${PAIR_SOURCE} \
#         -decode_algo greedy \
#     >>mol-interpolation.out 2>&1 & \
