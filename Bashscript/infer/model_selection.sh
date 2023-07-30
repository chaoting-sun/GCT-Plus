#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

# GPU_IDX=2
# BENCHMARK=moses
# MODEL_TYPE=cvaetf
# MODEL_NAME=${MODEL_TYPE}1
# EPOCH=40


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u   \
#     inference.py                                                   \
#         -use_cond2lat \
#     model-selection                                         \
#         -model_selection                                    \
#         -property_list logP tPSA QED                    \
#         -decode_algo multinomial                                   \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \
#         -n_jobs 16 \
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &


GPU_IDX=2
BENCHMARK=moses
MODEL_TYPE=vaetf
# MODEL_NAME=${MODEL_TYPE}2_
MODEL_NAME=${MODEL_TYPE}-warmup12000
# MODEL_NAME=${MODEL_TYPE}-warmup15000
EPOCH=40

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u   \
    inference.py                                                   \
        -use_cond2lat \
    model-selection                                         \
        -model_selection                                    \
        -decode_algo multinomial                                   \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -model_type ${MODEL_TYPE}                         \
        -model_name ${MODEL_NAME}                         \
        -epoch ${EPOCH}                                   \
        -n_jobs 16 \
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &


# GPU_IDX=0
# BENCHMARK=moses
# MODEL_TYPE=scacvaetfv3
# # MODEL_NAME=${MODEL_TYPE}-warmup15000
# # MODEL_NAME=${MODEL_TYPE}1
# # MODEL_NAME=${MODEL_TYPE}-beta0.01
# MODEL_NAME=${MODEL_TYPE}-beta0.01-warmup15000
# EPOCH=10


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u   \
#     inference.py                                                   \
#         -use_cond2lat \
#         -use_scaffold \
#     model-selection                                         \
#         -model_selection                                    \
#         -property_list logP tPSA QED                    \
#         -decode_algo multinomial                                   \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \
#         -n_jobs 16 \
#     >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &