#!/bin/python3
# Author: William Drew <wdrew@bwh.harvard.edu>

import numpy as np
from nilearn import image, maskers
from tqdm import tqdm
import argparse
import os
import time
import h5py
import csv

chunk_size = 200
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

parser = argparse.ArgumentParser(description = "Lesion Network Mapping using Precomputed Connectome")
parser.add_argument("-f", metavar='lesion_path', help="Path to CSV containing Lesion Paths or Path to Lesion Nifti", required=True)
parser.add_argument("-o", metavar='output_path', help="Path to Output Directory", required=True)
args = parser.parse_args()
lesion_path = args.f
output_path = args.o

if(os.path.isfile(lesion_path) == False):
    raise ValueError("Cannot find file: "+ lesion_path)
if(os.path.exists(output_path) == False):
    raise ValueError("Cannot find Output Directory: "+ output_path)

lesion_extension = lesion_path.split('/')[-1].split('.')[1:]

if 'csv' in lesion_extension:
    with open(lesion_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    lesion_paths_list = [path for line in data for path in line]
    print(f"Loaded {len(lesion_paths_list)} lesions...")
elif 'nii' in lesion_extension:
    lesion_paths_list = [lesion_path]
else:
    raise ValueError("Input File is not a NIfTI or a CSV containing paths to a list of NIfTIs")

# THESE PATHS ARE CURRENTLY HARDCODED, BUT SINCE WE ONLY HAVE A SINGLE PRECOMPUTED CONNECTOME
# FOR NOW, I'LL LEAVE IT LIKE THIS. IN THE FUTURE, WILL WANT TO STANDARDIZE FILE NAMES SO THAT
# WE WILL ONLY NEED TO SPECIFY A 'PRECOMPUTED CONNECTOME DIRECTORY'
mask_path = '/data/nimlab/precomputed_connectomes/masks/MNI152_T1_2mm_brain_mask_dil.nii.gz'
stdev_mask_path = '/data/nimlab/precomputed_connectomes/masks/MNI152_T1_2mm_brain_mask_dil_stdev_weighted.nii.gz'
connectome_R_path = '/data/nimlab/precomputed_connectomes/yeo1000_dil_AvgR.hdf5'
connectome_TC_path = '/data/nimlab/precomputed_connectomes/yeo1000_dil_TC.hdf5'
connectome_norm_path = '/data/nimlab/precomputed_connectomes/masks/yeo1000_dil_norms.npy'

# Load in all required connectomes
masker = maskers.NiftiMasker(mask_path).fit()
mask_size = int(np.sum(image.load_img(mask_path).get_fdata()))
connectome_norms = np.load(connectome_norm_path)
connectome_stdevs = masker.transform(stdev_mask_path)[0]
r = h5py.File(connectome_R_path, "r")
r_matrix = r['matrix']
tc = h5py.File(connectome_TC_path, "r")
tc_matrix = tc['matrix']
tc_num_points = tc_matrix.shape[1]

tic = time.perf_counter()

for lesion_path in tqdm(lesion_paths_list):
    lesion_name = lesion_path.split('/')[-1].split('.')[0]

    # Get lesioned Voxels
    lesion = masker.transform(lesion_path)[0]
    lesion_idx = np.where(lesion==1)[0]
    lesion_size = lesion_idx.shape[0]
    split_voxels = list(chunks(lesion_idx, chunk_size))

    # Calculate scaling factor and calculate weighted average R maps
    vector = np.zeros(tc_num_points)
    network_map_R = np.zeros(mask_size)
    total_weight = 0
    for section in split_voxels:
        vector += np.sum(tc_matrix[section,:], axis=0)
        weights = connectome_stdevs[section]
        total_weight += np.sum(weights)
        network_map_R += np.matmul(weights,r_matrix[section,:])
    denom = np.linalg.norm(vector)
    scaling_factor = np.sum(connectome_norms[lesion_idx])/denom
    r_network_map = network_map_R / total_weight

    # Scale average R maps and calculate Fz map
    r_final_map = r_network_map * scaling_factor
    fz_final_map = np.arctanh(r_final_map)

    # Output maps to file
    masker.inverse_transform(fz_final_map).to_filename(output_path + lesion_name + '_Precomputed_AvgR_Fz.nii.gz')
    masker.inverse_transform(r_final_map).to_filename(output_path + lesion_name + '_Precomputed_AvgR.nii.gz')

tc.close()
r.close()

print("Files output to: \n"+output_path)

toc = time.perf_counter()
print(f"Lesion Network Mapping completed in {toc - tic:0.4f} seconds ")
print(f"Time per lesion: {(toc - tic)/len(lesion_paths_list):0.4f} seconds ")

