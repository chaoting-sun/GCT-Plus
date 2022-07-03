#!/usr/bin/env bash 

nohup python3 -u \
    main_preprocess.py \
        -similarity 1 \
        -n_jobs 8 \
        -load_field \
        -data_name moses \
        -data_path /fileserver-gamma/chaoting/ML/dataset/moses/ \
        -field_path ./molGCT \
        -load_scaler \
        -variational \
        >preprocess.out 2>preprocess.err&