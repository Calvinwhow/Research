# #!/bin/bash

# # User-defined variables
# root_dir="/Users/cu135/Dropbox (Partners HealthCare)/resources/datasets/Queensland_PD_DBS_STN/BIDSdata/derivatives/leaddbs"
# intermediate_dir="/stimulations/native/gs_2023_Aysu/"
# file_pattern_1="*binary_model-simbio_hemi-L.nii"
# file_pattern_2="*binary_model-simbio_hemi-R.nii"
# output_file_pattern="binary_model-simbio_bl.nii"

# # Loop through each sub-directory
# for sub_dir in "${root_dir}"/sub-*; do
#     # Find files based on the patterns
#     file1=$(find "${sub_dir}${intermediate_dir}" -name "${file_pattern_1}" -type f)
#     file2=$(find "${sub_dir}${intermediate_dir}" -name "${file_pattern_2}" -type f)

#     # Check if both files exist
#     if [[ -f $file1 && -f $file2 ]]; then
#         # Define the output file path
#         output_file="${sub_dir}/stimulations/native/${output_file_pattern}"

#         # Add the two files together
#         fslmaths "$file1" -add "$file2" "$output_file"
#     else
#         echo "Missing file(s) in ${sub_dir}"
#     fi
# done
#!/bin/bash

# User-defined variables
root_dir="/Users/cu135/Dropbox (Partners HealthCare)/resources/datasets/Queensland_PD_DBS_STN/BIDSdata/derivatives/leaddbs/sub-*"
file_pattern_1="/stimulations/native/gs_2023Aysu/*binary_model-simbio_hemi-L.nii"
file_pattern_2="/stimulations/native/gs_2023Aysu/*binary_model-simbio_hemi-R.nii"
output_file_pattern="binary_model-simbio_bl.nii"

# Check if FSL is installed
if ! command -v fslmaths &> /dev/null; then
    echo "FSL (FMRIB Software Library) is not installed or not in your PATH."
    exit 1
fi

# Loop through each sub-directory
for sub_dir in "$root_dir"; do
    # Get a list of files matching file_pattern_1 and file_pattern_2
    files1=("${sub_dir}${file_pattern_1}")
    files2=("${sub_dir}${file_pattern_2}")

    # Check if both files exist
    if [ -e "${files1[0]}" ] && [ -e "${files2[0]}" ]; then
        # Define the output file path in the same directory as file1 and file2
        output_file="${files1[0]%/*}/${output_file_pattern}"

        # Add the two files together
        fslmaths "${files1[0]}" -add "${files2[0]}" "$output_file"
        echo "Savig file to : ${output_file}"
    else
        echo "Missing file(s) in ${sub_dir}"
        echo "${files1}"
        echo "${files2}"
    fi
done
