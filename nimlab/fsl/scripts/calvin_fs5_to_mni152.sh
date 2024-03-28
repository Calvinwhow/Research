#!/bin/bash
# This script processes brain imaging data within a a BIDS directory that was mounted to a Dockerfile.

ROOTDIR="/data" 

# Tell freesurfer where the subjects are
export SUBJECTS_DIR=/data

# Place FSAverage5 with the subjects
if [ ! -d "$ROOTDIR/fsaverage5" ]; then
    cp -r /opt/freesurfer-6.0.1/subjects/fsaverage5 $ROOTDIR/fsaverage5
fi

for dir in $ROOTDIR/*; do
    if [ -d "$dir" ]; then
        sub=$(basename "$dir")  # Assign subject identifier from the first field of the line
        # Skip the fsaverage5 directory
        if [ "$sub" == "fsaverage5" ]; then
            continue
        fi

        echo "Processing subject: $sub"
        # Make 'vol' directory
        if [ ! -d "$ROOTDIR/$sub/vol" ]; then
            mkdir -p "$ROOTDIR/$sub/vol"
        fi

        # Loop over hemispheres (left and right)
        for hemi in lh rh; do
            # Method A: convert fsaverage5 gii to nii, then nii to mni152 nii
            # Step 1: Convert the fsaverage space image into a volume image (still in fsaverage space)
            mri_surf2vol \
            --surfval "$ROOTDIR/$sub/surf/$hemi.${sub}_thickness.fs5.gii" \
            --identity "fsaverage5" \
            --template "$ROOTDIR/fsaverage5/mri/T1.mgz" \
            --hemi $hemi \
            --o $ROOTDIR/$sub/vol/$hemi.${sub}_vol_fsaverage.nii
            
            # Step 2: Convert the volume space in fsaverage space into MNI 152 space
            mri_vol2vol \
            --reg /opt/freesurfer-6.0.1/average/mni152.register.dat \
            --mov /opt/fsl-6.0.3/data/standard/MNI152_T1_2mm.nii.gz \
            --targ "$ROOTDIR/$sub/vol/$hemi.${sub}_vol_fsaverage.nii" \
            --inv \
            --interp nearest \
            --o "$ROOTDIR/$sub/vol/$hemi.${sub}_vol_MNI152.nii"
        done

        if [ -f "$ROOTDIR/$sub/vol/rh.${sub}_vol_MNI152.nii" ] && [ -f "$ROOTDIR/$sub/vol/lh.${sub}_vol_MNI152.nii" ]; then
            # merge both hemispheres into a single nifti
            fslmaths $ROOTDIR/$sub/vol/lh.${sub}_vol_MNI152.nii -add $ROOTDIR/$sub/vol/rh.${sub}_vol_MNI152.nii $ROOTDIR/$sub/vol/${sub}_vol_MNI152.nii
        else
            # One or both files do not exist
            echo "One or both hemisphere volume files are missing. Aborting hemisphere addition."
        fi
    fi
done
