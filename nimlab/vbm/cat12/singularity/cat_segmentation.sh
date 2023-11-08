#!/bin/bash

# Define the pattern to match your NIfTI files
NIFTI_PATH="/data/nimlab/dl_archive/adni_calvin/raws/*/*.nii"

#--PREPARE SINGULARITY
SINGULARITY_CONTAINER_URL='cat12-latest.sif docker://jhuguetn/cat12:latest'
singularity pull --name cat-r1853.simg $SINGULARITY_CONTAINER_URL

# Loop over the NIfTI files
for file in $NIFTI_PATH; do
  # Extract the directory path of the current file
  FILE_DIR=$(dirname "$file")
  # Extract the base name of the current file
  FILE_BASE=$(basename "$file")

  # Run the Singularity container with the dynamically specified paths
  singularity run --cleanenv --bind "$FILE_DIR":/mnt cat-r1853.simg \
    -b /batch/cat_standalone_segment.m \
    /mnt/"$FILE_BASE"
done
