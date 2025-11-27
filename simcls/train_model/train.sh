#!/bin/bash

dataset_dir="experiment/train/simcls-gen/candidate/cnn_dm/pegasus/preprocessed"

python SimCLS/main.py \
        --cuda \
        --gpuid 0 1\
        --dataset_dir $dataset_dir \
        -l 
