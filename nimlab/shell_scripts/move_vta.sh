#!/bin/bash

# Base directory where subdirectories are located
base_dir="/Volumes/OneTouch/resources/AD_dataset"

# Destination directory for the copied files
dest_dir="/Volumes/OneTouch/resources/AD_dataset/compound_vtas"

# Specific file we are looking for in the subdirectories
file_name="vat_compound_bl.nii"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through the first-level subdirectories of the base directory
for subdir in "$base_dir"/*/; do
    # Strip the base directory path and trailing slash to get the directory name
    dir_name=$(basename "$subdir")

    # Path of the target file in the current subdirectory
    target_file="${subdir}stimulations/MNI_ICBM_2009b_NLIN_ASYM/gs_20180403170745/$file_name"

    # Check if the target file exists
    if [ -f "$target_file" ]; then
        # Define the new file path with the directory name as prefix
        new_file="${dest_dir}/${dir_name}_${file_name}"

        # Copy the file to the new location
        cp "$target_file" "$new_file"
        echo "Copied $target_file to $new_file"
    else
        echo "File not found in $subdir"
    fi
done

echo "Operation completed."
