#!/bin/bash

# Relative path to Dockerfile
DOCKERFILE_PATH="vbm/cat12/docker/Dockerfile"

# Data path. Do not place wildcards here, but do place them in the WILCARDED_SUBJECT_PATH.
DATA_PATH="/data/nimlab/dl_archive/adni_calvin/raws"

# CAT12 script path
CAT12_SCRIPT_PATH="vbm/cat12/scripts/cat_12_vbm.sh"

# FSLMATHS script path
FSLMATHS_SCRIPT_PATH="vbm/cat12/scripts/fslmath_wm_gm_csf_tiv_correction.sh"

# Surface Extraction. Set to 1 if you want to extract surface values. Set to 0 if you do not. 
SURFACE_EXTRACTION=0

WILDCARDED_SUBJECT_PATH="*/*/sub*.nii*"

# Build the Docker image
docker build -t cat12-custom $DOCKERFILE_PATH

# Execute the Docker container and run your CAT12 script inside it
docker run --rm -it \
    -e "N_PARALLEL=8" \
    -e "TMP_DIR=/tmp" \
    -e "SURFACE_EXTRACTION=$SURFACE_EXTRACTION" \
    -e "WILDCARDED_SUBJECT_PATH=$WILDCARDED_SUBJECT_PATH" \
    -v $DATA_PATH:$DATA_PATH \
    -v $(dirname $CAT12_SCRIPT_PATH):/scripts \
    cat12-custom \
    /scripts/$(basename $CAT12_SCRIPT_PATH)

# Call FSLMATHS to perform post-processing Total Intracranial Volume Correction
bash $FSLMATHS_SCRIPT_PATH

