#!/bin/bash
# This script runs CAT12 standalone segmentation on a given folder of NIfTI files.

# Set the path to the CAT12.8.2_MCR standalone directory and MATLAB Runtime
CAT12_DIR=$CAT_PATH
MATLAB_RUNTIME=$MCRROOT

# 1) Run the CAT12 standalone segmentation script
cat_standalone.sh -b /opt/spm/standalone/cat_standalone_segment_calvin.m \
                    /data/*.nii*

# Wait for the segmentation to produce the output files
while [ $(ls /data/mri/*mwp*.nii 2> /dev/null | wc -l) -eq 0 ]; do
  echo "Waiting for segmentation to complete..."
  sleep 360
done

# 2) Run the CAT12 Standalone Smoothing 
cat_standalone.sh -b /opt/spm/standalone/cat_standalone_smooth_calvin.m \
                    /data/mri/*mwp*.nii

# 3) Extract TIV for each file. 
cat_standalone.sh -b /opt/spm/standalone/cat_standalone_get_TIV_calvin.m \
                    /data/report/cat_*.xml 