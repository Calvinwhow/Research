# #!/bin/bash
# Usage: easy_warp.sh
## Update the script and pass easy_reg.sh to the dockerfile with run_docker.py
# This script performs image warp on all pairs of reference and floating Nifti images in the /data directory.

# If you want to apply a deformation field to a different image (e.g., a segmentation, to propagate labels to another space), you can run:
# mri_easywarp --i <input_image> --o <output_image> --field <field> --threads <number_of_threads> --nearest
# where:

# <input_image>: (required) the image to deform.
# <output_image>: (required) the output (deformed) image.
# <field>: (required) the deformation field produced by mri_easyreg.
# <threads>: (optional) Number of threads to use (same as for mri_easyreg).
# <nearest>: (optional) Use this flag for nearest neighbor interpolation (default is trilinear). This is useful when deforming discrete segmentations.

# We note that the fields are specified in RAS, and therefore, can be used with images that do not necessarily live in the same voxel grid as those provided to mri_easyreg - but must of course live in the same real-world (RAS) coordinates.
cd /data

# Define the fixed reference image (MNI template). Specify the correct filename here.
output_image="warped_image_name" # Assuming segmentation of the reference is also available
field_image=""
threads=1
nearest=TRUE

# Find all floating images based on the naming convention (*.nii, excluding the reference and its segmentation)
input_images=$(find . -type f -name "*.nii" ! -name "$ref_image" ! -name "$ref_seg")

echo "***Currently available process limits:"
free -h
echo "***"

cat /sys/fs/cgroup/memory/memory.limit_in_bytes


# Loop through each floating image to process
for input_image in $input_image; do
    echo "Registration complete for: $flo_image."
done
