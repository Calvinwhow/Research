#!/bin/bash
# fslmath_tiv_correction_gm.sh

# RUNNING GM TIV CORRECTION
# Initialize a counter for the line number
LINE_NUMBER=1

# Path to the smoothed files
SMOOTHED_FILES_PATH="$DATA_PATH/CAT12.8.2/mri/mwp1sub*.nii"

# Divide each smoothed file by its corresponding TIV
for file in $SMOOTHED_FILES_PATH; do
  # Read the TIV value for the specific patient
  TIV=$(awk -v line=$LINE_NUMBER 'NR==line {print $1}' TIV.txt)
  
  # Perform the division
  fslmaths $file -div $TIV ${file}_divTIV
  
  # Increment the line number
  ((LINE_NUMBER++))
done

# RUNNING WM TIV CORRECTION
# Initialize a counter for the line number
LINE_NUMBER=1

# Path to the smoothed files
SMOOTHED_FILES_PATH="$DATA_PATH/CAT12.8.2/mri/mwp2sub*.nii"

# Divide each smoothed file by its corresponding TIV
for file in $SMOOTHED_FILES_PATH; do
  # Read the TIV value for the specific patient
  TIV=$(awk -v line=$LINE_NUMBER 'NR==line {print $1}' TIV.txt)
  
  # Perform the division
  fslmaths $file -div $TIV ${file}_divTIV
  
  # Increment the line number
  ((LINE_NUMBER++))
done

# RUNNING CSF TIV CORRECTION
# Initialize a counter for the line number
LINE_NUMBER=1

# Path to the smoothed files
SMOOTHED_FILES_PATH="$DATA_PATH/CAT12.8.2/mri/mwp3sub*.nii"

# Divide each smoothed file by its corresponding TIV
for file in $SMOOTHED_FILES_PATH; do
  # Read the TIV value for the specific patient
  TIV=$(awk -v line=$LINE_NUMBER 'NR==line {print $1}' TIV.txt)
  
  # Perform the division
  fslmaths $file -div $TIV ${file}_divTIV
  
  # Increment the line number
  ((LINE_NUMBER++))
done
