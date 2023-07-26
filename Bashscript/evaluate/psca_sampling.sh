#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


GPU_IDX=0
BENCHMARK=moses

# 沒跑過

MODEL_TYPE=scacvaetfv3
# MODEL_NAME=${MODEL_TYPE}-beta0.01-warmup15000
# MODEL_NAME=${MODEL_TYPE}-beta0.01-warmup15000
MODEL_NAME=${MODEL_TYPE}2-beta0.01-warmup15000_rand0.4
# MODEL_NAME=${MODEL_TYPE}-beta0.01-warmup15000

# MODEL_NAME=${MODEL_TYPE}-beta0.01-warmup15000_molgpt
EPOCH=20


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
        -use_cond2lat                               \
    psca-sampling                                   \
        -psca_sampling                              \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -decode_algo multinomial                                         \
        -model_type ${MODEL_TYPE}                                        \
        -model_name ${MODEL_NAME}                                        \
        -property_list logP tPSA QED                                     \
        -epoch ${EPOCH}                                                  \
        -sample_from test_scaffolds                                 \

    # >>psca_sampling-${MODEL_NAME}-${GPU_IDX}.out 2>&1 &

    # -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \

# GPU_IDX=3
# BENCHMARK=moses
# MODEL_TYPE=scacvaetfv3
# MODEL_NAME=${MODEL_TYPE}3-beta0.01-warmup15000_molgpt
# # MODEL_NAME=${MODEL_TYPE}2-beta0.01-warmup15000_rand0.2
# EPOCH=20


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u   \
#     inference.py                                                         \
#         -use_cond2lat                                                    \
#     psca-sampling                                                        \
#         -psca_sampling                                                   \
#         -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
#         -decode_algo multinomial                                         \
#         -model_type ${MODEL_TYPE}                                        \
#         -model_name ${MODEL_NAME}                                        \
#         -property_list logP tPSA SAS                                     \
#         -epoch ${EPOCH}                                                  \
#         -n_samples 10000                                                 \
#         -n_scaffolds 5                                                   \
#         -sample_from molgpt                                              \
#     # >>psca_sampling-${MODEL_NAME}-${GPU_IDX}.out 2>&1 &