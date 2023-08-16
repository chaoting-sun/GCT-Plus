#!/usr/bin/env bash

# export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
# export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

######################## vaetf ########################

# MODEL_TYPE=vaetf
# MODEL=${MODEL_TYPE}3
# EPOCH=37
# GPU_IDX=0

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#     mol-interpolation                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
#         -model_type vaetf                         \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}-${EPOCH} \
#         -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
#         -pair_source test_scaffolds \
#         -decode_algo greedy                        \
    # >>mol-interpolation_Model-${MODEL}_GPU-${GPU_IDX}.out 2>&1 & \

######################## pvaetf ######################## 

# MODEL_TYPE=pvaetf
# MODEL=${MODEL_TYPE}1
# EPOCH=15
# GPU_IDX=3

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py                                          \
#         -use_cond2lat                                     \
#     mol-interpolation                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
#         -property_list logP tPSA QED                      \
#         -model_type pvaetf                         \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}-${EPOCH} \
#         -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
#         -pair_source test_scaffolds \
#         -decode_algo greedy                        \
#     >>mol-interpolation_Model-${MODEL}_GPU-${GPU_IDX}.out 2>&1 & \

######################## scavaetf ########################

# MODEL_TYPE=scavaetf
# MODEL=${MODEL_TYPE}3
# EPOCH=16
# GPU_IDX=1

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py                                          \
#         -use_scaffold \
#     mol-interpolation                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
#         -model_type scavaetf                         \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}-${EPOCH} \
#         -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
#         -pair_source test_scaffolds \
#         -decode_algo greedy                        \
#     >>mol-interpolation_Model-${MODEL}_GPU-${GPU_IDX}.out 2>&1 & \

######################## pscavaetfv3 ########################


MODEL_TYPE=pscavaetf
MODEL=${MODEL_TYPE}3
EPOCH=17
GPU_IDX=1


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                          \
        -use_scaffold \
        -use_cond2lat                                     \
    mol-interpolation                                          \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/ \
        -property_list logP tPSA QED                      \
        -model_type pscavaetf                         \
        -model_name model_${EPOCH}.pt               \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}-${EPOCH} \
        -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
        -pair_source test_scaffolds \
        -decode_algo greedy                        \
    >>mol-interpolation_Model-${MODEL}_GPU-${GPU_IDX}.out 2>&1 & \


######################## ptf ########################

# BENCHMARK=moses

# MODEL_TYPE=ctf
# MODEL=${MODEL_TYPE}1
# EPOCH=30
# GPU_IDX=1

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                          \
#         -use_cond2lat                                     \
#     mol-interpolation                                          \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -property_list logP tPSA QED                      \
#         -model_type ctf                         \
#         -model_name model_${EPOCH}.pt               \
#         -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
#         -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/${MODEL}1-${EPOCH} \
#         -pair_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/mol-interpolation/ \
#         -pair_source test_scaffolds \
#         -decode_algo greedy                        \

