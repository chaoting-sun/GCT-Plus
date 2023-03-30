#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=0

# MODEL_TYPE=scacvaetfv1
# BENCHMARK=moses
# MODEL_NAME=${MODEL_TYPE}-s1.00
# EPOCH=12

BENCHMARK=moses
MODEL_TYPE=scacvaetfv2
MODEL_NAME=${MODEL_TYPE}-s1.00
EPOCH=20


# MODEL_TYPE=scacvaetfv2
# BENCHMARK=moses
# MODEL_NAME=${MODEL_TYPE}-s1.00-25ep-enum
# EPOCH=30


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
        -model_type ${MODEL_TYPE} \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
    scaffold-sampling                                  \
        -scaffold_sampling                         \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/${BENCHMARK}/ \
        -decode_algo multinomial                                         \
        -model_name ${MODEL_NAME}                                        \
        -epoch ${EPOCH}                        \
        -prop  4 70 0.894693                               \
    # >>scaffold_sampling-${MODEL_NAME}.out 2>&1 &