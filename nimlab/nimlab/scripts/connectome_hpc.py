# Author: Christopher Lin <clin5@bidmc.harvard.edu>


from nimlab import connectomics as cs
from nilearn import image, datasets, maskers
from glob import glob
import numpy as np
import argparse 
import os
import csv



def make_subjects(connectome_file_tuple, rois, masker):
    subjects = []
    roi_dict = {}
    for r in rois:
        roi_masked = masker.transform(r)
        roi_dict.update([(r , roi_masked)])
    f = connectome_file_tuple
    sub = cs.ConnectomeSubject(f[0],f[1], roi_dict)
    return sub



if __name__ == "__main__":
    # Make sure we don't aren't being a pest
    os.nice(19)
    # Get arguments
    parser = argparse.ArgumentParser(description = "Compute connectivity maps. Based off of LeadDBS's cs_fmri_conseed")
    parser.add_argument("-cs", metavar='connectome', help="SINGLE connecotme subject file", required=True)
    parser.add_argument("-r", metavar='ROI', help="ROI mask image", required=True)
    parser.add_argument("-min", metavar='input_mask', help="Mask with which to mask the input. Defaults to the MNI152_T1 brain. NOTE: The input mask MUST be the same as the mask used to generate the connectome files.")
    parser.add_argument("-mout", metavar='output_mask', help="Output mask. Defaults to no masking.")
    parser.add_argument("-c", metavar='command', help="Seed or ROI matrix. Defaults to seed.", default='seed')
    parser.add_argument("-o", metavar='output', help="Output directory for individual subject files. Should be in a scratch directory.", required=True)
    args = parser.parse_args()

    # Process arguments
    output_folder = args.o
    if(os.path.exists(output_folder) == False):
        raise ValueError("Invalid output folder: " + output_folder)
    if(args.min is None):
        mask_img = datasets.load_mni152_brain_mask()
    else:
        mask_img = image.load_img(args.min)
    brain_masker = maskers.NiftiMasker(mask_img)
    brain_masker.fit()

    if("nii" in args.r):
        roi_files = [args.r]
    else:
        roi_file_list = args.r
        roi_files = []
        flist = open(roi_file_list)
        reader = csv.reader(flist, delimiter = ',')
        for f in reader:
            roi_files.append(f[0])
        flist.close()
    
    connectome_file_norms = args.cs
    connectome_file = args.cs.split('_norms.npy')[0] + ".npy"
    subject = make_subjects((connectome_file, connectome_file_norms), roi_files, brain_masker)

    cs.make_fz_maps_to_file(subject, args.o)
