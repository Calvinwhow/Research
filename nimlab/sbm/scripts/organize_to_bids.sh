#!/bin/bash

# Get the directory from the command line argument
DIRECTORY="/Users/cu135/Dropbox (Partners HealthCare)/studies/atrophy_seeds_2023/shared_analysis/niftis_for_elmira/wmaps/sbm/6mm_w_map_final"

# Change to the specified directory
cd "$DIRECTORY" || exit

# Process files
for file in lh.* rh.*; do
    # Extract sub_id from the filename
    sub_id=$(echo "$file" | cut -d '.' -f 2)

    # Create a directory for the sub_id if it doesn't exist, within a 'sub-' prefix according to BIDS
    mkdir -p "sub-${sub_id}"

    # Copy the file into the corresponding sub_id directory, maintaining BIDS naming
    cp "$file" "sub-${sub_id}/$file"
done
