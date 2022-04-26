#!/usr/bin/env bash 

##########################################
########### Single-GPU Training ##########
##########################################

python3 -u \
        train.py -n_jobs 2 -variational -verbose \
        training --train_verbose

# CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 nohup python3 -u \
#                                               train.py -n_jobs 2 -variational -verbose \
#                                               training --train_verbose >train_test.out 2>train_test.err&


##########################################
########### Multi-GPU Training ###########
##########################################

# export NGPUs=2

##### multiple GPUs on a single node #####
# https://pytorch.org/docs/master/elastic/run.html#transitioning-from-torch-distributed-launch-to-torchrun

# CUDA_VISIBLE_DEVICES=1,2 torchrun \
#                          --standalone --nnodes 1 --nproc_per_node=2 \
#                          train.py -n_jobs 2 -variational -verbose \
#                          training --train_verbose \
                        #  >train_test.out 2>train_test.err&
                         

##### multiple GPUs on multiple nodes #####

# python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 \
#                          train.py -n_jobs 2 -variational -verbose \
#                          training --train_verbose \
#                          >train_test.out 2>train_test.err&
# 2 nodes are both in the 0th node.
