#!/bin/bash

# CAT12 script path
CAT12_SCRIPT_PATH="/PHShome/cu135/github_repository/Research/nimlab/vbm/cat12/scripts/cat_12_vbm.sh"

# FSLMATHS script path
FSLMATHS_SCRIPT_PATH="/PHShome/cu135/github_repository/Research/nimlab/vbm/cat12/scripts/fslmath_wm_gm_csf_tiv_correction.sh"

# Surface Extraction: 1 for yes, 0 for no
SURFACE_EXTRACTION=0

# Data path
DATA_PATH="/data/nimlab/dl_archive/adni_calvin/raws"

# Wildcard path to .nii files
WILDCARDED_SUBJECT_PATH="*/*/sub*.nii*"

# Environment variables for the Singularity container
export SINGULARITYENV_N_PARALLEL=8
export SINGULARITYENV_TMP_DIR=/tmp
export SINGULARITYENV_SURFACE_EXTRACTION=$SURFACE_EXTRACTION
export SINGULARITYENV_WILDCARDED_SUBJECT_PATH=$WILDCARDED_SUBJECT_PATH

# Run the Singularity container
singularity build cat12-latest.sif docker://jhuguetn/cat12:latest

singularity exec \
    -B $PWD:/data \
    -B $HOME/.matlab \
    -B $DATA_PATH:$DATA_PATH \
    -B $(dirname $CAT12_SCRIPT_PATH):/scripts \
    cat12-latest.sif \
    /scripts/$(basename $CAT12_SCRIPT_PATH)

# Pull the FSL container
singularity pull shub://aces/cbrain-containers-recipes:fsl_v6.0.1

# Run FSLMATHS for post-processing
singularity exec \
    -B $PWD:/data \
    -B $DATA_PATH:$DATA_PATH \
    -B $(dirname $FSLMATHS_SCRIPT_PATH):/scripts \
    fsl_v6.0.1.sif \
    /scripts/$(basename $FSLMATHS_SCRIPT_PATH)
    
# bash $FSLMATHS_SCRIPT_PATH
