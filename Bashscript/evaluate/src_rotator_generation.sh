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
        -model_type ctf \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
    src-rotator-generation                                  \
        -src_rotator_generation                             \
        -model_name ctf                    \
        -n_steps 1                                  \
        -epoch_list 30                       \
        -src_smiles 'Cn1ncc(Br)c1NC(=O)Nc1ccccc1' \
        -trg_props  2.82660 58.95 0.894693                 \
    # >>src-generation_model-${MODEL_NAME}.out 2>&1 &


## Cn1ncc(Br)c1NC(=O)Nc1ccccc1
## 2.82660 58.95 0.894693

## CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21
## 2.82212 57.61 0.883604

## O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2
## 2.84490 57.78 0.895593

## CC(C(O)c1ccccc1)N(C)C(=O)c1ncccc1Cl
## 2.92910 53.43 0.944642