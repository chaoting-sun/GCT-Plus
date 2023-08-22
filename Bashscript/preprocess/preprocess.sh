#!/usr/bin/env bash

# moses

python \
    preprocess.py \
        -save_folder ./data/ \
        -build_vocab \
        -scaled_properties logP tPSA QED \
    # > preprocess.out 2>&1 & 
