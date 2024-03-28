#!/bin/bash
#$1 = space delim sublist where line = sub sex age     (where sex is Male/Female)
#$2 = fwhm kernel size in mm
#$3 = subdir
#$4 = tmpdir
#$5 = glmdir
#$6 = outdir

fwhm=$2
while read f1 f2 f3; do
sub=$f1
    for hemi in lh rh; do
        #beta.mgh split should aready be done
        mri_surf2surf --srcsubject $sub --srcsurfval $3/$sub/surf/$hemi.thickness --trgsubject fsaverage5 --trgsurfval $4/$hemi.${sub}_thickness.fs5.$fwhm.mgh --fwhm $fwhm --cortex --hemi $hemi
        if [[ $f2 == "Male" ]]; then
            fscalc $5/$hemi/b0002.nii -mul $f3 -add $5/$hemi/b0000.nii --o $4/${hemi}_ct_expected.gii
        elif [[ $f2 == "Female" ]]; then
            fscalc $5/$hemi/b0002.nii -mul $f3 -add $5/$hemi/b0001.nii --o $4/${hemi}_ct_expected.gii
        else
        echo "second argument must equal 'Male' or 'Female'"
        exit 1
        fi
        fscalc $4/$hemi.${sub}_thickness.fs5.$fwhm.mgh -sub $4/${hemi}_ct_expected.gii --o $4/$hemi.thick_diff.gii
        fscalc $4/$hemi.thick_diff.gii -div $5/$hemi/rstd.mgh --o $4/${hemi}.${sub}.$fwhm.gii
        mv $4/${hemi}.${sub}.$fwhm.gii $6
        rm $4/$hemi.${sub}_thickness.fs5.$fwhm.mgh
        rm $4/${hemi}_ct_expected.gii
        rm $4/${hemi}.thick_diff.gii
    done
done < $1

chmod -R 770 $6