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


# augTransformer

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py                                    \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
    src-generation                                  \
        -src_generation                             \
        -model_name transformer_ep25_aug-all-s0.60-t0.10    \
        -n_steps 1                                  \
        -epoch_list 26                              \
        -src_smiles 'CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21' \
        -trg_props 2.82212 57.61 0.883604               \
    >>src-generation_model3.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py                                    \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
    src-generation                                  \
        -src_generation                             \
        -model_name transformer_ep25_aug-all-s0.70-t0.10    \
        -n_steps 1                                  \
        -epoch_list 26                              \
        -src_smiles 'CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21' \
        -trg_props 2.82212 57.61 0.883604               \
    >>src-generation_model2.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
    inference.py                                    \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
    src-generation                                  \
        -src_generation                             \
        -model_name transformer_ep25_aug-all-s0.80-t0.10    \
        -n_steps 1                                  \
        -epoch_list 26                              \
        -src_smiles 'CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21' \
        -trg_props 2.82212 57.61 0.883604               \
    >>src-generation_model3.out 2>&1 &

# Transformer

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py                                    \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat \
#     src-generation                                  \
#         -src_generation                             \
#         -model_name ${MODEL_NAME}1                  \
#         -n_steps 1                                  \
#         -epoch_list 20                              \
#         -src_smiles 'O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2' \
#         -trg_props  2.84490 57.78 0.895593                 \
#     >>src-generation_model-1.out 2>&1 &

# CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py                                    \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat \
#     src-generation                                  \
#         -src_generation                             \
#         -model_name ${MODEL_NAME}2                  \
#         -n_steps 1                                  \
#         -epoch_list 20                              \
#         -src_smiles 'O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2' \
#         -trg_props  2.84490 57.78 0.895593                 \
#     >>src-generation_model-1.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     inference.py                                    \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment_Repeat \
#     src-generation                                  \
#         -src_generation                             \
#         -model_name ${MODEL_NAME}3                  \
#         -n_steps 1                                  \
#         -epoch_list 20                              \
#         -src_smiles 'O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2' \
#         -trg_props  2.84490 57.78 0.895593                 \
#     >>src-generation_model-1.out 2>&1 &


## Cn1ncc(Br)c1NC(=O)Nc1ccccc1
## 2.82660 58.95 0.894693

## CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21
## 2.82212 57.61 0.883604

## O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2
## 2.84490 57.78 0.895593

## CC(C(O)c1ccccc1)N(C)C(=O)c1ncccc1Cl
## 2.92910 53.43 0.944642