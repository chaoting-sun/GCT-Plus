#!/usr/bin/env bash 

CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 nohup \
    python get_model.py training \
    >inference.out 2>inference.err &
