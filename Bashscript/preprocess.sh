#!/usr/bin/env bash 

nohup python3 -u \
    main_preprocess.py \
        -similarity 0.60 \
        -n_jobs 4 \
        -load_field \
        -data_name moses \
        -field_path ./molGCT \
        -load_scaler \
        -variational \
        >preprocess.out 2>preprocess.err&