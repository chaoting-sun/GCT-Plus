#!/usr/bin/env bash


export PYTHONPATH='/home/chaoting/tools/rdkit-tools/similarity/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/rdkit-tools/SMILES_plot/':$PYTHONPATH
export PYTHONPATH='/home/chaoting/tools/python-plot/':$PYTHONPATH



# CTF
# SIMILARITY=1.00
# TOLERANCE=0.00

# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python -u \
#     test.py                                         \
#         -model_type ctf \
#         -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
#     reconstruction                                  \
#         -reconstruction                             \
#         -model_name ctf                             \
#         -epoch_list 1 2 3 4 5 6 7 8 9 10                              \
#         -similarity ${SIMILARITY}                            \
#         -tolerance ${TOLERANCE}                             \
#     >>reconstruction_s${SIMILARITY}_t${TOLERANCE}.out 2>&1 &

        # -model_name ctf_aug-s${SIMILARITY}-t${TOLERANCE} \
        # -epoch_list 1 2 3 4 5 6 7 8 9 10            \


GPU_IDX=1
SIMILARITY=1.00
TOLERANCE=0.00
MODEL_NAME=ctf
# MODEL_NAME=cvaetfcut_aug-s1.00-t0.00_0.5

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python -u \
    test.py                                    \
        -model_type cvaetf \
        -train_path /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment \
    encoder_outputs                                     \
        -encoder_outputs                                \
        -model_name ${MODEL_NAME}                      \
        -epoch_list 10   \
        -similarity 1.00                               \
        -tolerance 0.00                                \
        -pad_to_same_len                               \
        
        # -debug                                         

    # >>rec_${MODEL_NAME}_s${SIMILARITY}_t${TOLERANCE}.out 2>&1 &

        # -epoch_list 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20                                 \
