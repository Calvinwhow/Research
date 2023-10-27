#!/bin/bash

# Get Dockerfile
docker pull jhuguetn/cat12

# Data path. Do not place wildcards here, but do place them in the WILCARDED_SUBJECT_PATH.
DATA_PATH="/data/nimlab/dl_archive/adni_calvin/raws"

# CAT12 script path
CAT12_SCRIPT_PATH="/PHShome/cu135/github_repository/Research/nimlab/vbm/cat12/scripts/cat_12_vbm.sh"

# FSLMATHS script path
FSLMATHS_SCRIPT_PATH="/PHShome/cu135/github_repository/Research/nimlab/vbm/cat12/scripts/fslmath_wm_gm_csf_tiv_correction.sh"

# Surface Extraction. Set to 1 if you want to extract surface values. Set to 0 if you do not. 
SURFACE_EXTRACTION=1

WILDCARDED_SUBJECT_PATH="*/*/sub*.nii*"

# Execute the Docker container and run your CAT12 script inside it
docker run --rm -it \
    -e "N_PARALLEL=8" \
    -e "TMP_DIR=/tmp" \
    -e "SURFACE_EXTRACTION=$SURFACE_EXTRACTION" \
    -e "WILDCARDED_SUBJECT_PATH=$WILDCARDED_SUBJECT_PATH" \
    -v $DATA_PATH:$DATA_PATH \
    -v $(dirname $CAT12_SCRIPT_PATH):/scripts \
    jhuguetn/cat12 \
    /scripts/$(basename $CAT12_SCRIPT_PATH)

# Pull the Docker image from Docker Hub
docker pull vistalab/fsl-v5.0

# Execute the Docker container and run your FSLMATHS script inside it
docker run --rm -it \
    -v $DATA_PATH:/input \
    -v $(dirname $FSLMATHS_SCRIPT_PATH):/scripts \
    vistalab/fsl-v5.0 \
    /scripts/$(basename $FSLMATHS_SCRIPT_PATH)
# Old Code to Run the TIV Correction
#bash $FSLMATHS_SCRIPT_PATH

