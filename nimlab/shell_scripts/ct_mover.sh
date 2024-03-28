#!/bin/bash

# Define the source and target base directories
source_base="/Volumes/One Touch/adni/EXPERIMENTALS"
target_base="/Users/cu135/Dropbox (Partners HealthCare)/studies/atrophy_seeds_2023/shared_analysis/niftis_for_elmira/atrophy_seeds"

# Find sub-{id} directories in the source base directory and iterate over them
find "${source_base}" -type d -name "sub-*" | while read sub_dir; do
    # Extract the sub-{id} part from the directory name
    sub_id=$(basename "${sub_dir}")
    
    # Define the unfixed target directory (with the original sub-id)
    unfixed_target_dir="${target_base}/${sub_id}/ses-01/thresholded_tissue_segment_z_scores"
    
    # Delete directories with the unfixed path name
    if [ -d "${unfixed_target_dir}" ]; then
        echo "Removing directory with unfixed path name: ${unfixed_target_dir}"
        rm -rf "${unfixed_target_dir}"
    fi

    # Remove the leading zero from the sub-id for the fixed path name
    sub_id_fixed=$(echo "${sub_id}" | sed 's/sub-0/sub-/')

    # Define the source and fixed target directories for thresholded_tissue_segment_z_scores
    source_dir="${sub_dir}/ses-01/thresholded_tissue_segment_z_scores"
    fixed_target_dir="${target_base}/${sub_id_fixed}/ses-01/thresholded_tissue_segment_z_scores"

    # Check if the source directory exists and the fixed target directory already exists
    if [ -d "${source_dir}" ] && [ -d "${fixed_target_dir}" ]; then
        echo "Source directory found: ${source_dir}"
        echo "Fixed target directory exists: ${fixed_target_dir}"

        # Find and copy *ct* files from the source to the fixed target directory
        find "${source_dir}" -type f -name "*ct*.nii" -exec cp {} "${fixed_target_dir}" \; -exec echo "Copied {} to ${fixed_target_dir}" \;
    else
        echo "Either source directory not found, or fixed target directory does not exist."
    fi
done

