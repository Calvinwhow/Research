#!/bin/python3
from nimlab import connectomics as cs
from multiprocessing import Pool
from nimlab import datasets as nimds
from nilearn import image, maskers
from tqdm import tqdm
from scipy.stats import pearsonr, ttest_1samp
from glob import glob
import pandas as pd
import argparse
import time
import os
import csv
import numpy as np


# Roi-roi correlation where correlation is computed between two sets of ROIs rather than the entire matrix.
# Apologies if this is a little hacky.
# Author: Christopher Lin <clin5@bidmc.harvard.edu>


class CorrMat:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# data_tuple is of the form (target_roi, rois, connectome) where target_roi is a 1D binary ROI mask, rois is a list of
# 1D binary ROI masks, and connectome is a 2D matrix (location x timecourse) for a single subject in the normative dataset
def compute_corrs(data_tuple):
    target_mask = data_tuple[0]
    roi_masks = data_tuple[1]
    connectome = data_tuple[2]

    target_tc = cs.extract_avg_signal(connectome, target_mask)
    roi_tcs = [cs.extract_avg_signal(connectome, r) for r in roi_masks]
    for r in roi_tcs:
        r[np.isnan(r)] = 0

    corrs = [pearsonr(target_tc, t)[0] for t in roi_tcs]
    return np.arctanh(np.asarray(corrs)).tolist()


# This is a mutator!
def corr_connectome_file(corrmat):
    connectome_vec = np.load(corrmat.connectome_file)
    corrs = []
    for t in corrmat.target_masks:
        fz_corr = compute_corrs((t,corrmat.roi_masks,connectome_vec))
        corrs.append(fz_corr)
    corrmat.corrs = corrs
    return corrmat






if __name__ == "__main__":
    os.nice(19)
    start = time.time()
    parser = argparse.ArgumentParser(description = "Compute connectivity from a list of ROIs to a list of target ROIs")
    parser = argparse.ArgumentParser(description = "Compute connectivity maps. Based off of LeadDBS's cs_fmri_conseed")
    parser.add_argument("-cs", metavar='connectome', help="Folder containing connectome/timecourse files", required=True)
    parser.add_argument("-r", metavar='ROI', help="CSV of ROI images", required=True)
    parser.add_argument("-t", metavar="targets", help="CSV of target ROI images", required = True)
    parser.add_argument("-bs", metavar='brain_space', help="Binary image that specifies which voxels of the brain to include in compressed timecourse files. Defaults to the MNI152_T1_2mm_brain_mask.nii.gz image included with FSL. NOTE: The brain_space must FIRST be used to generate the connectome files.")
    parser.add_argument("-o", metavar='output', help="Output directory", default='seed', required=True)
    parser.add_argument("-w", metavar='workers', help="Number of workers. Defaults to 12", type=int, default=int(12))
    args = parser.parse_args()
    if(args.bs is None):
        mask_img = nimds.get_img("MNI152_T1_2mm_brain_mask_dil")
    else:
        mask_img = image.load_img(args.bs)
    masker = maskers.NiftiMasker(mask_img, standardize=False).fit()

    # Read target files
    target_flist = open(args.t)
    target_files = []
    target_reader = csv.reader(target_flist, delimiter = ',')
    for f in target_reader:
        target_files.append(f[0])
    print("Reading Targets")
    target_vecs = [masker.transform(i) for i in target_files]
    target_flist.close()
    print("Targets read")

    # Read ROIs
    roi_flist = open(args.r)
    roi_files = []
    roi_reader = csv.reader(roi_flist, delimiter = ',')
    for f in roi_reader:
        roi_files.append(f[0])
    print("Reading ROIs")
    roi_vecs = [masker.transform(i) for i in tqdm(roi_files)]
    roi_flist.close()
    print("ROIs read")

    # Get connectome files
    connectome_files_norms = glob(args.cs + "/*_norms.npy")
    connectome_files = [glob(f.split('_norms')[0] + ".npy")[0] for f in connectome_files_norms]
    if (len(connectome_files) == 0):
        raise ValueError("No connectome files found")

    # Create matrix objects (that will be averaged at end
    corrmats = []
    for i in range(0, len(connectome_files)):
        corrmat = CorrMat(roi_masks=roi_vecs,target_masks=target_vecs,
                connectome_file=connectome_files[i],corrs=[])
        corrmats.append(corrmat)
    p = Pool(args.w)
    print("Calculating correlations")
    corrmats = list(tqdm(p.imap(corr_connectome_file,corrmats),total=len(corrmats)))
    p.close()


    # Construct avg fz correlations
    roi_corr_mat = np.zeros((len(target_files),len(roi_files),len(connectome_files)))
    for i in range(0, len(target_files)):
        for j in range(0, len(roi_files)):
            roi_corrs = [x.corrs[i][j] for x in corrmats]
            roi_corr_mat[i,j,:] = np.arctanh(roi_corrs)

    avg_fz_df = pd.DataFrame()
    avg_fz_df['roi'] = roi_files
    for t in range(0, len(target_files)):
        avg_fz_df[target_files[t]] = np.mean(roi_corr_mat[t,:,:],axis=1)


    # Construct avg r
    r_df = pd.DataFrame()
    r_df['roi'] = roi_files
    for t in target_files:
        r_df[t] = avg_fz_df[t].apply(lambda x: np.tanh(x))

    # Construct t
    t_df = pd.DataFrame()
    t_df['roi'] = roi_files
    for t in range(0, len(target_files)):
        t_vals = ttest_1samp(roi_corr_mat[t,:,:], 0, axis = 1).statistic
        t_df[target_files[t]] = t_vals




    # Output files
    avg_fz_df.to_csv(args.o + '/fz_corrs.csv', header = True)
    r_df.to_csv(args.o + '/r_corrs.csv', header = True)
    t_df.to_csv(args.o + '/t_corrs.csv', header = True)

    end = time.time()
    elapsed = end - start
    print("Total elapsed: " + str(elapsed))
    print("Avg time per roi-target pair: " + str(elapsed/(len(roi_files) * len(target_files))))
