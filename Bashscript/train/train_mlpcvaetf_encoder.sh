#!/usr/bin/env bash


GPU_IDX=2

CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
    train.py \
        -model_type mlpcvaetf_encoder \
        -tolerance 0.00  \
        -similarity 0.80 \
        -start_epoch 1   \
        -num_epoch 30    \
        -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlpcvaetf_encoder \
    # >train_transformer-140.out 2>&1 &


# train
# CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     main_train.py \
#         -similarity 1 \
#         -n_jobs 2 \
#         -load_field \
#         -field_path /fileserver-gamma/chaoting/ML/cvae-transformer/molGCT/fields/ \
#         -load_scaler \
#         -variational \
#     train-2nd \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim1 \
#         -batch_size 128 \
#         -num_epoch 40 \
#         -starting_epoch 1 \
#         -train_verbose \
#         -train_stage 2 \
#     >train_sim1.out 2>train_sim1.err &

# MODEL_TYPE='mlp_encoder'
# SIMILARITY=0.70
# GPU_IDX=3
# NUM_EPOCH=40
# START_EPOCH=1
# LOSS_FCN=kld
# BATCH_SIZE=128


# train
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 python3 -u \
# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     main_train.py \
#         -similarity ${SIMILARITY} \
#         -n_jobs 2 \
#         -load_field \
#         -load_scaler \
#         -variational \
#     train-2nd \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${MLP_STACK} \
#         -batch_size 128 \
#         -num_epoch ${NUM_EPOCH} \
#         -starting_epoch 1 \
#         -train_verbose \
#         -train_stage 2 \
#     >train_sim${SIMILARITY}_${MLP_STACK}.out 2>train_sim${SIMILARITY}_${MLP_STACK}.err &


# CUDA_VISIBLE_DEVICES=${GPU_IDX} CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#     main_train.py \
#         -similarity ${SIMILARITY} \
#         -loss_fcn ${LOSS_FCN} \
#         -n_jobs 2 \
#         -load_field \
#         -load_scaler \
#         -model_type ${MODEL_TYPE} \
#         -variational \
#     train-2nd \
#         -save_directory /fileserver-gamma/chaoting/ML/cvae-transformer/Experiment/mlptf_train_stage2_sim${SIMILARITY}_${LOSS_FCN} \
#         -batch_size ${BATCH_SIZE} \
#         -num_epoch ${NUM_EPOCH} \
#         -start_epoch ${START_EPOCH} \
#         -train_verbose \
#         -train_stage 2 \
#     >train_sim${SIMILARITY}_${LOSS_FCN}.out 2>train_sim${SIMILARITY}_${LOSS_FCN}.err &
