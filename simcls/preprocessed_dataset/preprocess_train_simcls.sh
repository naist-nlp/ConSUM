#!/bin/bash

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    INPUT_FILE=experiment/train/simcls-gen/candidate/cnn_dm/bart/candidate_16.json
    OUTPUT_DIR=experiment/train/simcls-gen/candidate/cnn_dm/bart/preprocessed
    DATASET_NAME=cnn_dm
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    INPUT_FILE=experiment/train/simcls-gen/candidate/cnn_dm/pegasus/candidate_16.json
    OUTPUT_DIR=experiment/train/simcls-gen/candidate/cnn_dm/pegasus/preprocessed
    DATASET_NAME=cnn_dm
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    INPUT_FILE=experiment/train/simcls-gen/candidate/cnn_dm/t5-large/candidate_16.json
    OUTPUT_DIR=experiment/train/simcls-gen/candidate/cnn_dm/t5-large/preprocessed
    DATASET_NAME=cnn_dm
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    INPUT_FILE=experiment/train/simcls-gen/candidate/xsum/bart/candidate_16.json
    OUTPUT_DIR=experiment/train/simcls-gen/candidate/xsum/bart/preprocessed
    DATASET_NAME=xsum
elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    INPUT_FILE=experiment/train/simcls-gen/candidate/xsum/pegasus/candidate_16.json
    OUTPUT_DIR=experiment/train/simcls-gen/candidate/xsum/pegasus/preprocessed
    DATASET_NAME=xsum
elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    INPUT_FILE=experiment/train/simcls-gen/candidate/xsum/t5-large/candidate_16.json
    OUTPUT_DIR=experiment/train/simcls-gen/candidate/xsum/t5-large/preprocessed
    DATASET_NAME=xsum
else
    echo "Unknown SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"  # Optional: Set a default log file path for other cases
    exit 0
fi

echo "SLURM_ARRAY_JOB_ID" $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID" $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_TASK_COUNT" $SLURM_ARRAY_TASK_COUNT
echo "SLURM_ARRAY_TASK_MAX" $SLURM_ARRAY_TASK_MAX
echo "SLURM_ARRAY_TASK_MIN" $SLURM_ARRAY_TASK_MIN

python SimCLS/custom_data_preprocess.py \
        --input_file $INPUT_FILE \
        --output_dir $OUTPUT_DIR \
        --dataset_name $DATASET_NAME