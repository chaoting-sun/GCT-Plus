#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=2
# MODEL_NAME=transformer_aug-s0.50-t0.10
MODEL_NAME=transformer

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py                                          \
#     src-generation                                        \
#         -src_generation                                   \
#         -n_steps 1 2 3 4                                  \
#         -model_name ${MODEL_NAME}                         \
#         -epoch_list 20                                    \
#         -src_smiles 'CC(C(O)c1ccccc1)N(C)C(=O)c1ncccc1Cl' \
#         -trg_props  2.92910 53.43 0.944642                \
#     >>src-generation_model-${MODEL_NAME}.out 2>&1 &

# src: 2.92910 53.43 0.944642


CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat \
    src-generation                                  \
        -src_generation                             \
        -model_name ${MODEL_NAME}3                   \
        -n_steps 1 2 3 4                                  \
        -epoch_list 20                              \
        -src_smiles 'CC(C(O)c1ccccc1)N(C)C(=O)c1ncccc1Cl' \
        -trg_props  2.92910 53.43 0.944642                     \
    # >>src-generation_model-${MODEL_NAME}.out 2>&1 &

