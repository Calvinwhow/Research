#!/bin/bash
#$1 = fsgd file path
#$2 = glmdir
#$3 = FS SUBJECTS_DIR path
#$4 = tmpdir
#$5 = fwhm kernel size in mm

export SUBJECTS_DIR=$3

for hemi in lh rh; do
    mris_preproc --fsgd $1 --target fsaverage5 --hemi $hemi --meas thickness --out $4/$hemi.model.thickness.mgh
    mri_surf2surf --hemi $hemi --s fsaverage5 --srcsurfval $4/$hemi.model.thickness.mgh --fwhm $5 --cortex --trgsurfval $4/$hemi.model.thickness.$5.mgz
    mri_glmfit --y $4/$hemi.model.thickness.$5.mgz --fsgd $1 doss --glmdir $2/$hemi --surf fsaverage5 $hemi --no-contrasts-ok
    mri_convert $2/$hemi/beta.mgh $2/$hemi/b.nii --split
done