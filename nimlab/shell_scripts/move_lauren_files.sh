#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="/Users/cu135/Dropbox (Partners HealthCare)/AD_dataset"
DEST_DIR="/Users/cu135/Dropbox (Partners HealthCare)/for_lauren"

# Iterate over all sub-directories in the source directory
for subdir in "$SOURCE_DIR"/*; do
    if [[ -d "$subdir" ]]; then
        # Extract the patient number from the directory name
        patient_num=$(basename "$subdir")

        # Define the source file and the destination directory for this patient
        source_file="$subdir/ea_reconstruction.mat"
        dest_patient_dir="$DEST_DIR/$patient_num"

        # Check if the source file exists
        if [[ -f "$source_file" ]]; then
            # Create the destination directory for this patient
            mkdir -p "$dest_patient_dir"
            # Copy the source file to the destination directory
            cp "$source_file" "$dest_patient_dir"
        fi
    fi
done
