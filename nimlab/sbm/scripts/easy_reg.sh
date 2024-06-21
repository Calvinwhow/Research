# #!/bin/bash
# Usage: easy_reg.sh
## Update the script and pass easy_reg.sh to the dockerfile with run_docker.py
# This script performs image registration on all pairs of reference and floating Nifti images in the /data directory.

# You can use EasyReg with the following command:
# mri_easyreg --ref <reference_image> --flo <floating_image>  \
#             --ref_seg <ref_image_segmentation> --flo_seg <flo_image_segmentation>  \
#             --ref_reg [deformed_ref_image] --flo_reg <deformed_flo_image>  \
#             --fwd_field [forward_field] --bak_field <backward_field>  \
#             --threads <number_of_threads> --affine_only

# where:
# <reference_image>: (required) the reference image in .nii(.gz) or .mgz format (note that, since the method is symmetric, the choice of reference vs floating is arbitrary).
# <floating_image>: (required) the floating image.
# <ref_image_segmentation>: (required) file with the SynthSeg v2 (non-robust) segmentation + parcellation of the reference image. If it does not exist, EasyReg will create it. If it already exists (e.g., from a previous EasyReg run), then EasyReg will read it from disk (which is faster than segmenting).
# <flo_image_segmentation>: (required) same for floating image.
# <deformed_ref_image>: (optional) this is the file where the deformed (registered) reference image is written.
# <deformed_flo_image>: (optional) this is the file where the deformed (registered) floating image is written.
# <forward_field>: (optional) this is the file where the forward deformation field is written. The deformation includes both the affine and nonlinear components. Must be a nifti (.nii/.nii.gz) or .mgz file; it is encoded as the real world (RAS) coordinates of the target location for each voxel.
# <backward_field>: (optional) this is the file where the backward deformation field is written. It must also be a nifty or mgz file.
# <threads>: (optional) Number of threads to use. Set to -1 to use the maximum (i.e., the total number of available cores). Note that the default is 1, which will run much more slowly.
# --affine_only: (optional) Use this flag to skip the nonlinear registration estimated by the neural network (i.e., do affine only).

cd /data

# Define the fixed reference image (MNI template). Specify the correct filename here.
ref_image="icbm_avg_152_t1_tal_nlin_symmetric_VI.nii"
ref_seg=reference_segment.nii # Assuming segmentation of the reference is also available

# Find all floating images based on the naming convention (*.nii, excluding the reference and its segmentation)
flo_images=$(find . -type f -name "*.nii" ! -name "$ref_image" ! -name "$ref_seg")

echo "***Currently available process limits:"
free -h
echo "***"

cat /sys/fs/cgroup/memory/memory.limit_in_bytes


# Loop through each floating image to process
for flo_image in $flo_images; do
    echo "Processing floating image: $flo_image with reference image: $ref_image..."
    
    # Extract base name for output files
    base_name="${flo_image%.*}"

    # Assume segmentation for the floating image is named similarly
    flo_seg="${base_name}_seg.nii"  # Update this if segmentation naming differs

    # Define output file names
    deformed_ref_image="${base_name}_deformed_ref.nii"
    deformed_flo_image="${base_name}_deformed_flo.nii"
    forward_field="${base_name}_forward_field.nii"
    backward_field="${base_name}_backward_field.nii"

    # Run registration using mri_easyreg
    mri_easyreg --ref "$ref_image" --flo "$flo_image" \
                --ref_seg "$ref_seg" --flo_seg "$flo_seg" \
                --ref_reg "$deformed_ref_image" --flo_reg "$deformed_flo_image" \
                --fwd_field "$forward_field" --bak_field "$backward_field" \
                --threads 1  # Modify as needed

    echo "Registration complete for: $flo_image."
done
