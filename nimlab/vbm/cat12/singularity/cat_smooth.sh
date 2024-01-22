#!/bin/bash

#This needs to find each unique base name. Then, it must smooth each existing mwp1,2,3. 

#--PREPARE SINGULARITY
SINGULARITY_CONTAINER_URL='cat12-latest.sif docker://jhuguetn/cat12:latest'
singularity pull --name cat-r1853.simg $SINGULARITY_CONTAINER_URL

# Define the pattern to match your NIfTI files
NIFTI_PATH="/data/nimlab/dl_archive/adni_calvin/raws/all_pts"

# Loop over the NIfTI files
for file in $NIFTI_PATH; do
  # Extract the directory path of the current file
  FILE_DIR=$(dirname "$file")
  # Extract the base name of the current file
  FILE_BASE=$(basename "$file")

  echo "Initiating File: $FILE_BASE"
  # Run the Singularity container with the dynamically specified paths
  singularity run --cleanenv --bind "$FILE_DIR":/mnt cat-r1853.simg \
    -b /batch/cat_standalone_smooth.m \
    /data/enigma-data/raw/CAT12.8.2/mri/mwp1sub*.nii  \
    /mnt/"$FILE_BASE" -a1 "[6 6 6]" -a2 "'s6'"
done
