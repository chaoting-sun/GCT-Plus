#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=2
BENCHMARK=moses


# MODEL_TYPE=vaetf
# MODEL_NAME=${MODEL_TYPE}
# EPOCH=30


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#     encoder-test                                          \
#         -encoder_test                                     \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \

    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &


MODEL_TYPE=cvaetf
MODEL_NAME=${MODEL_TYPE}3
EPOCH=25


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                          \
        -use_cond2lat                                     \
    encoder-test                                          \
        -property_list logP tPSA QED                      \
        -encoder_test                                     \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -model_type ${MODEL_TYPE}                         \
        -model_name ${MODEL_NAME}                         \
        -epoch ${EPOCH}                                   \

#     >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &


# MODEL_TYPE=scacvaetfv3
# MODEL_NAME=${MODEL_TYPE}
# EPOCH=15

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_scaffold                                     \
#         -use_cond2lat                                     \
#     encoder-test                                          \
#         -property_list logP tPSA QED                      \
#         -encoder_test                                     \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \



# MODEL_TYPE=ctf
# MODEL_NAME=${MODEL_TYPE}1
# EPOCH=30

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_cond2lat                                     \
#     encoder-test                                          \
#         -property_list logP tPSA QED                      \
#         -encoder_test                                     \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \

