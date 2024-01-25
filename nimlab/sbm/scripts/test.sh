
ROOTDIR="/data" 

# Tell freesurfer where the subjects are
export SUBJECTS_DIR=/data

# Place FSAverage5 with the subjects
if [ ! -d "$ROOTDIR/fsaverage5" ]; then
    cp -r /opt/freesurfer-6.0.1/subjects/fsaverage5 $ROOTDIR/fsaverage5
fi


ls /opt/freesurfer-6.0.1/average