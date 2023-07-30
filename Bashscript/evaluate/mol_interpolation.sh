#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

######################## vaetf ########################

# BENCHMARK=moses

# MODEL_TYPE=vaetf
# MODEL=${MODEL_TYPE}1_
# EPOCH=38
# GPU_IDX=0

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#     mol-interpolation                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type vaetf                         \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}-${EPOCH} \
#         -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
#         -pair_source test_scaffolds \
#         -decode_algo greedy                        \

######################## cvaetf ######################## 

# BENCHMARK=moses

# MODEL_TYPE=cvaetf
# MODEL=${MODEL_TYPE}1
# EPOCH=14
# GPU_IDX=0

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_cond2lat                                     \
#     mol-interpolation                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -property_list logP tPSA QED                      \
#         -model_type cvaetf                         \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}1-${EPOCH} \
#         -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
#         -pair_source test_scaffolds \
#         -decode_algo greedy                        \

######################## scavaetf ########################

# BENCHMARK=moses

# MODEL_TYPE=scavaetf
# MODEL=${MODEL_TYPE}1-warmup15000
# EPOCH=16
# GPU_IDX=1

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_scaffold \
#     mol-interpolation                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type scavaetf                         \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}-${EPOCH} \
#         -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
#         -pair_source test_scaffolds \
#         -decode_algo greedy                        \

######################## scacvaetfv3 ########################

BENCHMARK=moses

MODEL_TYPE=scacvaetfv3
MODEL=${MODEL_TYPE}1-beta0.01-warmup15000
EPOCH=17
GPU_IDX=1

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                          \
        -use_scaffold \
        -use_cond2lat                                     \
    mol-interpolation                                          \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -property_list logP tPSA QED                      \
        -model_type scacvaetfv3                         \
        -model_name model_${EPOCH}.pt               \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}1-${EPOCH} \
        -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
        -pair_source test_scaffolds \
        -decode_algo greedy                        \


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