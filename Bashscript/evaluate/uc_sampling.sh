#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=2
EPOCH=40
MODEL=vaetf1_

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u                                                           \
    inference.py                                                                                                           \
    uc-sampling                                                                                                            \
        -decode_algo multinomial                                                                                           \
        -save_folder ${SAVE_FOLDER}                                                                                        \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/                                                   \
        -model_type vaetf                                                                                                  \
        -model_name model_${EPOCH}.pt                                                                                      \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL}/                    \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/uc-sampling/${MODEL}-${EPOCH}/ \
        -n_jobs 8                                                                                                          \
        -n_samples 1000
    # >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &



# MODEL_TYPE=vaetf
# MODEL_NAME=${MODEL_TYPE}3
# EPOCH=42
# 30 35


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u   \
#     inference.py                                                   \
#     unconditioned-sampling                                         \
#         -unconditioned_sampling                                    \
#         -decode_algo multinomial                                   \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -model_type ${MODEL_TYPE}                         \
#         -model_name ${MODEL_NAME}                         \
#         -epoch ${EPOCH}                                   \
#         -n_jobs 16 \
#     >>${MODEL_NAME}_${GPU_IDX}.out 2>&1 &