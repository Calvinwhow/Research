#!/bin/bash

# Path to config.json
CONFIG_PATH="./config.json"  # Update this to the actual relative path where config.json is located in your setup

# Read from config.json
CAT12_IMAGE=$(jq -r '.docker_images.cat12' $CONFIG_PATH)
FSL_IMAGE=$(jq -r '.docker_images.fsl' $CONFIG_PATH)
DATA_PATH=$(jq -r '.user_input.data_path' $CONFIG_PATH)
CAT12_SCRIPT_PATH=$(jq -r '.paths.cat12_script_path' $CONFIG_PATH)
FSLMATHS_SCRIPT_PATH=$(jq -r '.paths.fslmaths_script_path' $CONFIG_PATH)
SURFACE_EXTRACTION=$(jq -r '.user_input.surface_extraction' $CONFIG_PATH)
WILDCARDED_SUBJECT_PATH=$(jq -r '.user_input.wildcarded_subject_path' $CONFIG_PATH)
N_PARALLEL=$(jq -r '.settings.n_parallel' $CONFIG_PATH)
TMP_DIR=$(jq -r '.settings.tmp_dir' $CONFIG_PATH)

# Pull the Cat12 Docker image from Docker Hub
docker pull $CAT12_IMAGE

# Execute the Docker container and run your CAT12 script inside it
docker run --rm -it \
    -e "N_PARALLEL=$N_PARALLEL" \
    -e "TMP_DIR=$TMP_DIR" \
    -e "SURFACE_EXTRACTION=$SURFACE_EXTRACTION" \
    -e "WILDCARDED_SUBJECT_PATH=$WILDCARDED_SUBJECT_PATH" \
    -v $DATA_PATH:$DATA_PATH \
    -v $(dirname $CAT12_SCRIPT_PATH):/scripts \
    $CAT12_IMAGE \
    /scripts/$(basename $CAT12_SCRIPT_PATH)

# Pull the FSL Docker image from Docker Hub
docker pull $FSL_IMAGE

# Execute the Docker container and run your FSLMATHS script inside it
docker run --rm -it \
    -v $DATA_PATH:/input \
    -v $(dirname $FSLMATHS_SCRIPT_PATH):/scripts \
    $FSL_IMAGE \
    /scripts/$(basename $FSLMATHS_SCRIPT_PATH)