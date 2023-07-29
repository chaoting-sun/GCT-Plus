#!/usr/bin/env bash

export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH


GPU_IDX=1
EPOCH=16
MODEL=scavaetf3-warmup15000


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u         \
    inference.py                                                         \
        -use_cond2lat                                                    \
    sca-sampling                                                         \
        -data_folder /fileserver-gamma/chaoting/ML/dataset/moses/        \
        -decode_algo multinomial                                         \
        -model_type scavaetf                                             \
        -model_name model_${EPOCH}.pt                                    \
        -model_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/moses/${MODEL} \
        -save_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling/${MODEL}-${EPOCH} \
        -scaffold_folder /fileserver-gamma/chaoting/ML/cvae-transformer/Inference-Dataset/moses/sca-sampling \
        -n_samples 10000                                                   \
        -scaffold_source train                                      \
        # -use_molgpt
        # -substructure
    # >>sca_sampling-${MODEL_NAME}-${GPU_IDX}.out 2>&1 & 


    # -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment-Dataset/${BENCHMARK}/${MODEL_NAME} \
