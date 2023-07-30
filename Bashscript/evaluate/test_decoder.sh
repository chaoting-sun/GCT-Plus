#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


BENCHMARK=moses

######################## sca-vaetf ########################

# MODEL_TYPE=scavaetf
# MODEL_NAME=${MODEL_TYPE}1-warmup15000
# EPOCH=15
# GPU_IDX=0

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_cond2lat                                     \
#         -use_scaffold                                     \
#     decoder-test                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -decode_algo greedy                        \
#         -epoch ${EPOCH}                                   \


######################## vaetf ########################

# MODEL_TYPE=vaetf
# MODEL_NAME=${MODEL_TYPE}1_
# EPOCH=37
# GPU_IDX=0

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#     decoder-test                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -decode_algo greedy                        \
#         -epoch ${EPOCH}                                   \

######################## cvaetf ######################## 

MODEL_TYPE=cvaetf
MODEL_NAME=${MODEL_TYPE}1
EPOCH=14
GPU_IDX=2

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                          \
        -use_cond2lat                                     \
    mol-interpolation                                          \
        -property_list logP tPSA QED                      \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -model_type ${MODEL_TYPE}                         \
        -model_name ${MODEL_NAME}                         \
        -epoch ${EPOCH}                                   \
#     # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &

######################## scacvaetfv3 ########################

# GPU_IDX=3
# MODEL_TYPE=scacvaetfv3
# MODEL_NAME=${MODEL_TYPE}1-beta0.01-warmup15000
# EPOCH=17

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


# GPU_IDX=3
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