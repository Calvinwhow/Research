#!/bin/bash

# This script performs VBM for all tissue segments: 

# Set the number of parallel processes
N_PARALLEL=${N_PARALLEL:-8}  # Use default value of 8 if not set

# Set temporary directory for log files
TMP_DIR=${TMP_DIR:-/tmp}  # Use default value of /tmp if not set

# Path to the CAT12.8.2_MCR standalone
CAT12_DIR="/software/CAT12.8.2_MCR"

# Path to Matlab Runtime
MATLAB_RUNTIME="$CAT12_DIR/v93"

# Path to your raw nifti data
RAW_DATA_PATH=${DATA_PATH}

# Set Wildcard Pathing to Subjects
WILDCARDED_SUBJECT_PATH=${WILDCARDED_SUBJECT_PATH:-"/sub*.nii*"}

# Skip surface extractison by setting output.surface to 0. Engage by setting to 1. 
SURFACE_EXTRACTION=${SURFACE_EXTRACTION:-0} # Use default value of 0 if not set

# Prepare Directories
mkdir -p $TMP_DIR

# Segment and preprocess the data (this also implicitly does warping)
SEGMENT_CMD="-m $MATLAB_RUNTIME -b $CAT12_DIR/standalone/cat_standalone_segment_enigma.m -a \"matlabbatch{1}.spm.tools.cat.estwrite.output.surface = $SURFACE_EXTRACTION;\" $RAW_DATA_PATH/$WILDCARDED_SUBJECT_PATH"

# Run segmentation in parallel
$CAT12_DIR/standalone/cat_parallelize.sh -p $N_PARALLEL -l $TMP_DIR -c "$SEGMENT_CMD"

# After segmentation, you'll have mwp1* files for gray matter, which are modulated and warped. 
# Let's smooth these files. Recommended smoothing size is 6mm.

# Prepare smoothing command
SMOOTH_CMD="-m $MATLAB_RUNTIME -b $CAT12_DIR/standalone/cat_standalone_smooth.m $RAW_DATA_PATH/CAT12.8.2/mri/mwp1sub*.nii -a1 \"[6 6 6]\" -a2 \"'s6'\""

# Run smoothing in parallel
$CAT12_DIR/standalone/cat_parallelize.sh -p $N_PARALLEL -l $TMP_DIR -c "$SMOOTH_CMD"

# Estimate and Save Total Intra-cranial Volume (TIV)
TIV_CMD="-m $MATLAB_RUNTIME -b $CAT12_DIR/standalone/cat_standalone_get_TIV.m $RAW_DATA_PATH/CAT12.8.2/report/cat_*.xml -a1 \"'TIV.txt'\" -a2 \"1\" -a3 \"0\""

# Run TIV estimation in parallel
$CAT12_DIR/standalone/cat_parallelize.sh -p $N_PARALLEL -l $TMP_DIR -c "$TIV_CMD"
