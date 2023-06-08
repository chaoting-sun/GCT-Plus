#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=2
BENCHMARK=moses


######################## vaetf ########################

# MODEL_TYPE=vaetf
# MODEL_NAME=${MODEL_TYPE}1_
# EPOCH=37

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#     decoder-test                                          \
#         -decoder_test                                     \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -decode_algo greedy                        \
#         -epoch ${EPOCH}                                   \

######################## cvaetf ######################## 

MODEL_TYPE=cvaetf
MODEL_NAME=${MODEL_TYPE}1
EPOCH=21

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                          \
        -use_cond2lat                                     \
    decoder-test                                          \
        -property_list logP tPSA QED                      \
        -decoder_test                                     \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -model_type ${MODEL_TYPE}                         \
        -model_name ${MODEL_NAME}                         \
        -epoch ${EPOCH}                                   \
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &

######################## scacvaetfv3 ########################

# MODEL_TYPE=scacvaetfv3
# MODEL_NAME=${MODEL_TYPE}-warmup15000
# EPOCH=3

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_scaffold                                     \
#         -use_cond2lat                                     \
#     decoder-test                                          \
#         -property_list logP tPSA QED                      \
#         -decoder_test                                     \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \

#     >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &

        # -decode_algo multinomial                          \
        # -top_k 2                                          \



######################## ctf ########################


# MODEL_TYPE=ctf
# MODEL_NAME=${MODEL_TYPE}1
# EPOCH=30


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_cond2lat                                     \
#     decoder-test                                          \
#         -property_list logP tPSA QED                      \
#         -decoder_test                                     \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &

        # -decode_algo multinomial                          \
        # -top_k                                           \