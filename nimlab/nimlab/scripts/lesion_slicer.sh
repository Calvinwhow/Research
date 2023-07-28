#!/bin/bash
#set -x
# Created by Alex Cohen to take slices from lesion masks

. /etc/fsl/5.0/fsl.sh

# list_of_lesions=$1
i=$1

# for i in list_of_lesions; do

    read X Y Z <<<$(fslstats $i -c)
    read x y z <<<$(fslstats $i -C)
    fslroi $i tempx $x 1 -1 -1 -1 -1
    fslroi $i tempy -1 -1 $y 1 -1 -1
    fslroi $i tempz -1 -1 -1 -1 $z 1
    
    flirt -in tempx -ref $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz -applyxfm -usesqform -out full_tempx
    flirt -in tempy -ref $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz -applyxfm -usesqform -out full_tempy
    flirt -in tempz -ref $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz -applyxfm -usesqform -out full_tempz
    
    X=$( printf "%.0f" $X )
    Y=$( printf "%.0f" $Y )
    Z=$( printf "%.0f" $Z )

    x=$( printf "%.0f" $x )
    y=$( printf "%.0f" $y )
    z=$( printf "%.0f" $z )


    a=`basename $i`
    fslmaths full_tempx -max full_tempy -max full_tempz Slice_${X}_${Y}_${Z}_of_${a}
#    echo Making Slice_${X}_${Y}_${Z}_of_${a}
    rm *temp*
# done
