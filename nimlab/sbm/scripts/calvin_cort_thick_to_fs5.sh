#!/bin/bash
# This script processes brain imaging data within a a BIDS directory that was mounted to a Dockerfile.

for dir in /opt/freesurfer-6.0.1/subjects/*; do
    if [ -d "$dir" ]; then
        sub=$(basename "$dir")  # Assign subject identifier from the first field of the line
        
        echo "In dir: $sub"
        echo "Processing subject: $sub"

        if test -f "/opt/freesurfer-6.0.1/subjects/$sub/surf/$hemi.thickness"; then
            echo "The file exists."
        else
            echo "The file does not exist."
        fi

        # Loop over hemispheres (left and right)
        for hemi in lh rh; do
            # Process thickness data for the subject and hemisphere using mri_surf2surf
            mri_surf2surf --srcsubject $sub --srcsurfval /opt/freesurfer-6.0.1/subjects/$sub/surf/$hemi.thickness --trgsubject fsaverage5 --trgsurfval /opt/freesurfer-6.0.1/subjects/$hemi.${sub}_thickness.fs5.gii --hemi $hemi
        done
    fi
done