#!/bin/bash
# $1 = lh gifti file path, should have filename lh.<subid>.gii
# $2 = rh gifti file path, should have filename rh.<subid>.gii
# $3 = FreeSurfer fsaverage space ("fsaverage", "fsaverage5", "fsaverage6")
# $4 = output directory

if [ "$#" -ne 4 ]; then
  echo "Error: Incorrect number of arguments. Expecting 4, got $#" >&2
  exit 1
fi

freesurfer_subject=$3

# Make output directory if it doesn't exist
if [ ! -d "$4" ]; then
    mkdir -p "$4"
fi

lh_fname=$(basename "$1")
rh_fname=$(basename "$2")

if [[ "$lh_fname" != lh.* ]]; then
    echo "Error: left hemisphere filename does not begin with 'lh.'" >&2
    exit 1
elif [[ "$rh_fname" != rh.* ]]; then
    echo "Error: right hemisphere filename does not begin with 'rh.'" >&2
    exit 1
elif [[ ! -d "$FREESURFER_HOME/subjects/$freesurfer_subject" ]]; then
    echo "Error: Freesurfer subject $freesurfer_subject does not exist" >&2
    exit 1
else
    lh_subid=$(echo "$lh_fname" | cut -d'.' -f2)
    rh_subid=$(echo "$rh_fname" | cut -d'.' -f2)

    if [ "$lh_subid" = "$rh_subid" ]; then
        mkdir $4/tmp_vol
        mkdir $4/tmp_mni
        
        mri_surf2vol --surfval $1 --identity $freesurfer_subject --template $FREESURFER_HOME/subjects/$freesurfer_subject/mri/T1.mgz --hemi lh --o $4/tmp_vol/lh.${lh_subid}_vol.nii
        mri_vol2vol --inv --targ $4/tmp_vol/lh.${lh_subid}_vol.nii --mov $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz --o $4/tmp_mni/lh.${lh_subid}.nii --interp nearest --reg $FREESURFER_HOME/average/mni152.register.dat

        mri_surf2vol --surfval $2 --identity $freesurfer_subject --template $FREESURFER_HOME/subjects/$freesurfer_subject/mri/T1.mgz --hemi rh --o $4/tmp_vol/rh.${rh_subid}_vol.nii
        mri_vol2vol --inv --targ $4/tmp_vol/rh.${rh_subid}_vol.nii --mov $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz --o $4/tmp_mni/rh.${rh_subid}.nii --interp nearest --reg $FREESURFER_HOME/average/mni152.register.dat

        fslmaths $4/tmp_mni/lh.${lh_subid}.nii -add $4/tmp_mni/rh.${rh_subid}.nii $4/${lh_subid}.nii

        rm -r $4/tmp_vol/
        rm -r $4/tmp_mni/
    else
        echo "Error: Surface hemisphere subject names do not match!" >&2
        exit 1
    fi
fi
echo "DONE!"