#!/bin/bash

# Get the directory of this script
SCRIPT_DIR=$(dirname "$0")

# Relative path to Dockerfile
DOCKERFILE_PATH="/Users/cu135/Library/CloudStorage/OneDrive-Personal/OneDrive_Documents/Work/Software/Research/nimlab/vbm/cat12/containers/Dockerfile"

# CAT12 script path
CAT12_SCRIPT_PATH="$SCRIPT_DIR/cat_12_vbm.sh"

# FSLMATHS script path
FSLMATHS_SCRIPT_PATH="$SCRIPT_DIR/fslmath_wm_gm_csf_tiv_correction.sh"

# NIFTI File Directory
DATA_PATH="/Users/cu135/Dropbox (Partners HealthCare)/resources/datasets/ADNI/neuroimaging/raws"

# String to Find Subjects from the NIFTI File Directory
WILDCARDED_SUBJECT_PATH="*/*/sub*.nii*"

# Surface Extraction. Set to 1 if you want to extract surface values. Set to 0 if you do not. 
SURFACE_EXTRACTION=0



#----------------------------------------------------------------DO NOT TOUCH----------------------------------------------------------------

#---- USING THE PULL
docker pull jhuguetn/cat12
docker run --rm -it \
    -e "N_PARALLEL=8" \
    -e "TMP_DIR=/tmp" \
    -e "SURFACE_EXTRACTION=$SURFACE_EXTRACTION" \
    -e "WILDCARDED_SUBJECT_PATH=$WILDCARDED_SUBJECT_PATH" \
    -v $DATA_PATH:$DATA_PATH \
    -v $(dirname $CAT12_SCRIPT_PATH):/scripts \
    jhuguetn/cat12 \
    /scripts/$(basename $CAT12_SCRIPT_PATH)

#---- USING MY DOCKERFILE
# Build the Docker image

# docker build -t cat12vbmdocker "$DOCKERFILE_PATH"
# # Execute the Docker container and run your CAT12 script inside it
# docker run --rm -it \
#     -e "N_PARALLEL=8" \
#     -e "TMP_DIR=/tmp" \
#     -e "SURFACE_EXTRACTION=$SURFACE_EXTRACTION" \
#     -e "WILDCARDED_SUBJECT_PATH=$WILDCARDED_SUBJECT_PATH" \
#     -v $DATA_PATH:$DATA_PATH \
#     -v $(dirname $CAT12_SCRIPT_PATH):/scripts \
#     cat12vbmdocker \
#     /scripts/$(basename $CAT12_SCRIPT_PATH)

# Call FSLMATHS to perform post-processing Total Intracranial Volume Correction
bash $FSLMATHS_SCRIPT_PATH

