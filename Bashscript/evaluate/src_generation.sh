#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH

GPU_IDX=0
MODEL_NAME=cvaetf
# MODEL_NAME=cvaetf_aug-s0.70-t0.20

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    inference.py                                    \
        -model_type cvaetf \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
    src-generation                                  \
        -src_generation                        \
        -decode_algo multinomial \
        -model_name ${MODEL_NAME}                             \
        -n_steps 2                                  \
        -epoch_list 20                              \
        -src_smiles 'CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21' \
        -trg_props 2.82660 58.95 0.894693               \


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
#     inference.py                                    \
#         -model_type cvaetf \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
#     src-generation                                  \
#         -src_generation                        \
#         -decode_algo multinomial \
#         -model_name ${MODEL_NAME}                             \
#         -n_steps 1                                  \
#         -epoch_list 20                              \
#         -src_smiles 'CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21' \
#         -trg_props 2.82660 58.95 0.894693               \
    # >/dev/null 2>&1 &

# -trg_props 2.82212 57.61 0.883604               \

# 1, 6, 0.1
# -trg_props1 3.82212 57.61 0.883604               \
# -trg_props2 2.82212 63.61 0.883604               \
# -trg_props3 2.82212 57.61 sh               \

# src: 2.92910 53.43 0.944642


## Cn1ncc(Br)c1NC(=O)Nc1ccccc1
## 2.82660 58.95 0.894693

## CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21
## 2.82212 57.61 0.883604

## O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2
## 2.84490 57.78 0.895593

## CC(C(O)c1ccccc1)N(C)C(=O)c1ncccc1Cl
## 2.92910 53.43 0.944642