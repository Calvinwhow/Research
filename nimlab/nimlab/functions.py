from __future__ import print_function

import os
import glob
import json
import shutil
import fnmatch
import warnings
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import multiprocessing

from numpy import ma
from scipy import stats
from random import random
from bids import BIDSLayout
from pymongo import MongoClient
from nimlab import datasets as nimds
from nimlab import surface as nimsf
from nimlab import configuration as config
from IPython.core.getipython import get_ipython
from fsl.wrappers import flirt, fslmaths, wrapperutils
from nilearn import plotting, image, regions, datasets, maskers, surface

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import logging

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

""" Miscellanious helpful functions.

Credits: Alex Cohen, Louis Soussand, Christopher Lin, and William Drew

"""

BRAIN_MASK = nimds.get_img("MNI152_T1_2mm_brain_mask_dil")


@wrapperutils.fslwrapper
def fsl_denan(src, out, **kwargs):
    """De-NaN a .nii/.nii.gz file.

    Args:
        src (path to .nii/.nii.gz file): File to de-NaN
        ref (path to .nii/.nii.gz file): De-NaN'ed output file path

    Returns:
        tuple : ('', '')
    """
    cmd = ["fslmaths", src, "-nan", out]
    return cmd + wrapperutils.applyArgStyle("-=", **kwargs)


def fsl_reslice_to_standard(src, standard, out):
    """Reslice a .nii/.nii.gz file to a given standard template with fsl_denan and flirt (fslpy). Uses Spline interpolation.

    Args:
        src (path to .nii/.nii.gz file): File to reslice
        standard (Niimg-like object): Standard template to reslice to
        out (path to .nii/.nii.gz file): Resliced output file path
    """
    source_path = os.path.abspath(src)
    out_path = os.path.abspath(out)
    fsl_denan(src=source_path, out=out_path)
    flirt(
        src=out_path,
        ref=standard,
        out=out_path,
        usesqform=True,
        applyxfm=True,
        interp="spline",
    )
    fslmaths(out_path).bin().run(out_path)
    fsl_denan(src=out_path, out=out_path)


# Helper function to find a pattern within a directory tree
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# Helper function to find the longest common substring between two strings
# Source (archive): https://archive.fo/vg52E
# Source: https://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php
def lcs(S, T):
    m = len(S)
    n = len(T)
    counter = [[0] * (n + 1) for x in range(m + 1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i + 1][j + 1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i - c + 1 : i + 1])
                elif c == longest:
                    lcs_set.add(S[i - c + 1 : i + 1])

    return lcs_set


def cbig_vol2surf(lesion_list, output_dir, threshold=0.5, surf_space="fsaverage5"):
    """Convert a list of Niftis to Giftis using CBIG Registration Fusion
    * DEPRECATED
    * Author: William Drew (wdrew@bwh.harvard.edu)

    - Intended to be used from within connectome_quick.py in an sshpass call with
      erisone_wrapper.sh to setup env variables

    - Requires Matlab (use module) and Freesurfer (at /data/nimlab/software)

    Args:
        lesion_list (list): List of paths to .nii/.nii.gz files.

        output_dir (str): Path to output directory.

        threshold (float): Threshold for masking projected Volume to Surface ROIS. Defaults to 0.5.

        surf_space (str): Freesurfer fsaverage surface mesh to use. Defaults to fsaverage5

                            - 'fsaverage3': the low-resolution fsaverage3 mesh (642 nodes)

                            - 'fsaverage4': the low-resolution fsaverage4 mesh (2562 nodes)

                            - 'fsaverage5': the low-resolution fsaverage5 mesh (10242 nodes)

                            - 'fsaverage6': the medium-resolution fsaverage6 mesh (40962 nodes)

                            - 'fsaverage': the high-resolution fsaverage mesh (163842 nodes)

    Returns:
        list of lists of output converted volume-to-surface file paths

        Format:
                [ [/path/to/subject1/lh_roi.gii,/path/to/subject1/rh_roi.gii],
                  [/path/to/subject2/lh_roi.gii,/path/to/subject2/rh_roi.gii],
                  [/path/to/subject3/lh_roi.gii,/path/to/subject3/rh_roi.gii],
                                                 .
                                                 .
                                                 .                            ]

    Outputs:
        For each ROI given in lesion_list, generates two surface files (left/right hemisphere)

    """
    config.verify_software(["freesurfer_path", "cbig_path"])
    freesurfer_mesh_dict = {
        "fsaverage3": "3",
        "fsaverage4": "4",
        "fsaverage5": "5",
        "fsaverage6": "6",
        "fsaverage": "7",
    }

    output_path_list = []

    # Setup Environment Variables
    cmd = [
        "MATLAB_BIN_PATH=$(which matlab)",
        "&&",
        "export MATLAB_BIN_PATH=${MATLAB_BIN_PATH%/matlab}",
    ]
    cmd.append(f"export FREESURFER_HOME={config.software['freesurfer_path']}")
    cmd.append("&&")
    cmd.append("export FSFAST_HOME=$FREESURFER_HOME/fsfast")
    cmd.append("&&")
    cmd.append("export SUBJECTS_DIR=$FREESURFER_HOME/subjects")
    cmd.append("&&")
    cmd.append("export MNI_DIR=$FREESURFER_HOME/mni")
    cmd.append("&&")
    cmd.append("source $FREESURFER_HOME/SetUpFreeSurfer.sh")
    cmd.append("&&")
    cmd.append("username=$USER")
    cmd.append("&&")
    cmd.append("firstCharacter=${username::1}")
    cmd.append("&&")
    cmd.append("export TMPDIR=/scratch/$firstCharacter/$username")

    ipython = get_ipython()
    ipython.system(" ".join(cmd))

    for lesion in lesion_list:
        roi_name = os.path.basename(lesion).split(".")[0]

        cmd = [
            f"bash {config.software['cbig_path']}/registration/standalone_scripts_for_MNI_fsaverage_projection/CBIG_RF_projectMNI2fsaverage.sh",
            "-m $MATLAB_BIN_PATH",
            "-s",
        ]
        # Project volume to surface space with CBIG Registration Fusion
        cmd.append(os.path.abspath(lesion))
        cmd.append("-o")
        cmd.append(os.path.abspath(output_dir))
        cmd.append("&&")

        # Probably making some bad assumptions about the output of this script but...
        full_texture_left = os.path.join(
            output_dir,
            "lh." + roi_name + "_.allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz",
        )
        full_texture_right = os.path.join(
            output_dir,
            "rh." + roi_name + "_.allSub_RF_ANTs_MNI152_orig_to_fsaverage.nii.gz",
        )

        # and Convert CBIG Surface to Surface mesh of choice using FreeSurfer mri_surf2surf

        cmd.append(f"{config.software['freesurfer_path']}/bin/mri_surf2surf")
        cmd.append("--hemi lh")
        cmd.append("--srcsubject ico")
        cmd.append("--srcsurfval")
        cmd.append(os.path.abspath(full_texture_left))
        cmd.append("--trgsubject ico")
        cmd.append("--trgicoorder")
        cmd.append(freesurfer_mesh_dict[surf_space])
        cmd.append("--trgsurfval")
        converted_texture_left = os.path.abspath(
            full_texture_left[:-7] + freesurfer_mesh_dict[surf_space] + ".nii.gz"
        )
        cmd.append(converted_texture_left)

        cmd.append("&&")

        cmd.append(f"{config.software['freesurfer_path']}/bin/mri_surf2surf")
        cmd.append("--hemi rh")
        cmd.append("--srcsubject ico")
        cmd.append("--srcsurfval")
        cmd.append(os.path.abspath(full_texture_right))
        cmd.append("--trgsubject ico")
        cmd.append("--trgicoorder")
        cmd.append(freesurfer_mesh_dict[surf_space])
        cmd.append("--trgsurfval")
        converted_texture_right = os.path.abspath(
            full_texture_right[:-7] + freesurfer_mesh_dict[surf_space] + ".nii.gz"
        )
        cmd.append(converted_texture_right)

        # Run Bash Scripts
        ipython.system(" ".join(cmd))

        # Reshape Converted CBIG Surface to pseudo Gifti format
        texture_left = image.get_data(converted_texture_left)
        texture_right = image.get_data(converted_texture_right)

        texture_left_mask = np.reshape(
            (texture_left > threshold).astype(int), (-1,), order="F"
        )
        texture_left_mask = nimsf.new_gifti_image(data=texture_left_mask)
        texture_left_mask.to_filename(output_dir + "lh." + roi_name + ".gii")

        texture_right_mask = np.reshape(
            (texture_right > threshold).astype(int), (-1,), order="F"
        )
        texture_right_mask = nimsf.new_gifti_image(data=texture_right_mask)
        texture_right_mask.to_filename(output_dir + "rh." + roi_name + ".gii")

        os.remove(converted_texture_left)
        os.remove(converted_texture_right)

        output_path_list.append(
            [
                output_dir + "lh." + roi_name + ".gii",
                output_dir + "rh." + roi_name + ".gii",
            ]
        )

    return output_path_list


def nilearn_vol2surf(lesion_list, output_dir, threshold=0.5, surf_space="fsaverage5"):
    """Convert a list of Niftis to Pseudo Giftis using Nilearn's surface.vol_to_surf
    * DEPRECATED
    * Author: William Drew (wdrew@bwh.harvard.edu)

    Args:
        lesion_list (list): List of paths to .nii/.nii.gz files.

        output_dir (str): Path to output directory.

        threshold (float): Threshold for masking projected Volume to Surface ROIS. Defaults to 0.5.

        surf_space (str): Freesurfer fsaverage surface mesh to use. Defaults to fsaverage5

                            - 'fsaverage3': the low-resolution fsaverage3 mesh (642 nodes)

                            - 'fsaverage4': the low-resolution fsaverage4 mesh (2562 nodes)

                            - 'fsaverage5': the low-resolution fsaverage5 mesh (10242 nodes)

                            - 'fsaverage6': the medium-resolution fsaverage6 mesh (40962 nodes)

                            - 'fsaverage': the high-resolution fsaverage mesh (163842 nodes)

    Returns:
        list of lists of output converted volume-to-surface file paths

        Format:
                [ [/path/to/subject1/lh_roi.nii.gz,/path/to/subject1/rh_roi.nii.gz],
                  [/path/to/subject2/lh_roi.nii.gz,/path/to/subject2/rh_roi.nii.gz],
                  [/path/to/subject3/lh_roi.nii.gz,/path/to/subject3/rh_roi.nii.gz],
                                                 .
                                                 .
                                                 .                                   ]

    Outputs:
        For each ROI given in lesion_list, generates two surface files (left/right hemisphere )
    """
    fsaverage = datasets.fetch_surf_fsaverage(mesh=surf_space)
    output_path_list = []

    for lesion in lesion_list:
        texture_left = surface.vol_to_surf(
            lesion,
            surf_mesh=fsaverage.pial_left,
            inner_mesh=fsaverage.white_left,
            interpolation="linear",
        )
        texture_right = surface.vol_to_surf(
            lesion,
            surf_mesh=fsaverage.pial_right,
            inner_mesh=fsaverage.white_right,
            interpolation="linear",
        )

        roi_name = os.path.basename(lesion).split(".")[0]

        texture_left_mask = np.reshape(
            (texture_left > threshold).astype(int), (-1,), order="F"
        )
        texture_left_mask = nimsf.new_gifti_image(data=texture_left_mask)
        texture_left_mask.to_filename(output_dir + "lh." + roi_name + ".gii")

        texture_right_mask = np.reshape(
            (texture_right > threshold).astype(int), (-1,), order="F"
        )
        texture_right_mask = nimsf.new_gifti_image(data=texture_right_mask)
        texture_right_mask.to_filename(output_dir + "rh." + roi_name + ".gii")

        output_path_list.append(
            [
                output_dir + "lh." + roi_name + ".gii",
                output_dir + "rh." + roi_name + ".gii",
            ]
        )

    return output_path_list


def sum_imgs(imgs):
    """Sum together a list of images.

    Args:
        imgs (list of Niimg-like objects): Images to sum

    Returns:
        Niimg-like object: Summed image
    """
    sum_dat = np.zeros(imgs[0].shape)
    for i in imgs:
        sum_dat += i.get_fdata()
    sum_img = image.new_img_like(imgs[0], sum_dat)
    return sum_img


def overlap_maps(imgs, t, mask="MNI152_T1_2mm_brain_mask_dil"):
    """Threshold and binarize maps at a particular t-stat level (both negative and positive), and then sum
    them together to produce a map of overlaps.

    Args:
        imgs (list of Niimg-like objects): Images to overlap
        t (float): t-statistic threshold
        mask (str): name of binary mask in nimlab.datasets or path to binary mask to mask output maps with

    Returns:
        Niimg-like: Overlap map
    """
    if nimds.check_mask(mask):
        mask_img = nimds.get_img(mask)
    else:
        mask_img = image.load_img(mask)
    pos_imgs = [image.math_img(f"(img > {t})*mask", img=i, mask=mask_img) for i in imgs]
    neg_imgs = [
        image.math_img(f"(img < -{t})*mask", img=i, mask=mask_img) for i in imgs
    ]
    return sum_imgs(pos_imgs), sum_imgs(neg_imgs)


def plot_overlap_map_to_screen(imgs, t, fractions, mask="MNI152_T1_2mm_brain_mask_dil"):
    """Create a visualization of overlaps at a particular T value, thresholded
    at a particular overlap fraction.

    Args:
        imgs (list of Niimg-like objects): Images to overlap
        t (float): t-statistic threshold
        fractions ([type]): Percentage of overlap to threshold at
        mask (str): name of binary mask in nimlab.datasets or path to binary mask to mask output maps with
    """
    pos_overlap, neg_overlap = overlap_maps(imgs, t, mask)
    combined_overlap = image.math_img("img1 - img2", img1=pos_overlap, img2=neg_overlap)
    combined_fraction = image.math_img(
        f"np.divide(img1, {len(imgs)})", img1=combined_overlap
    )
    # Plot an unthresholded map
    plotting.plot_stat_map(
        combined_fraction,
        display_mode="z",
        cut_coords=list(range(-54, 72, 6)),
        cmap="Spectral_r",
        colorbar=True,
        title="Unthresholded",
    )
    for i in fractions:  # note the i-(0.5/N) to get >= instead of > thresholding
        plotting.plot_stat_map(
            combined_fraction,
            display_mode="z",
            cut_coords=list(range(-54, 72, 6)),
            cmap="Spectral_r",
            colorbar=True,
            threshold=i - 0.5 / len(imgs),
            title=f"Thresholded at >= {i}",
        )


def write_overlap_map_to_file(imgs, t, outdir, mask="MNI152_T1_2mm_brain_mask_dil"):
    """Overlap a set of maps and at a t-level and then output them to files.

    Args:
        imgs (list of Niimg-like): Images to overlap
        t (float): t-stat level to threshold at
        outdir (str): Path to output images
        mask (str): name of binary mask in nimlab.datasets or path to binary mask to mask output maps with
    """
    pos_overlap, neg_overlap = overlap_maps(
        imgs, t, mask="MNI152_T1_2mm_brain_mask_dil"
    )
    combined_overlap = image.math_img("img1 - img2", img1=pos_overlap, img2=neg_overlap)
    combined_fraction = image.math_img(
        f"np.divide(img1, {len(imgs)})", img1=combined_overlap
    )
    combined_overlap.to_filename(os.path.join(outdir, f"LNM_overlap_at_T-{t}.nii.gz"))
    combined_fraction.to_filename(os.path.join(outdir, f"LNM_fraction_at_T-{t}.nii.gz"))


def define_ROIs_from_overlap(
    imgs, t, fraction, minimum_region_size, mask="MNI152_T1_2mm_brain_mask_dil"
):
    """Overlap a set of maps at a t-stat level, threshold at an overlap fraction level,
    then extract the connected regions to use as ROIs for other analyses.

    Args:
        imgs (list of Niimg-like): Maps to overlap
        t (float): t-stat threshoold
        fraction (float): fraction threshold
        minimum_region_size (int): Minimum size to count as a potential ROI
        mask (str): name of binary mask in nimlab.datasets or path to binary mask to mask output maps with

    Returns:
        peak_regions (Niimg-like): 4d image, where each slice in the 4th dimension contains
            a single peak region
        roi_index (ndarray): Array of labels which correspond to peak region indices
    """
    pos_overlap, neg_overlap = overlap_maps(imgs, t, mask)
    combined_overlap = image.math_img("img1 - img2", img1=pos_overlap, img2=neg_overlap)
    combined_fraction = image.math_img(
        f"np.divide(img1, {len(imgs)})", img1=combined_overlap
    )
    thresholded_fraction = image.threshold_img(
        combined_fraction, fraction - 0.5 / len(imgs)
    )
    peak_regions, roi_index = regions.connected_regions(
        thresholded_fraction,
        extract_type="connected_components",
        min_region_size=minimum_region_size,
    )
    return peak_regions, roi_index


def hotBlues(posthres=0, negthres=0, reverse=False):
    # colors1a = plt.cm.YlGnBu_r(np.linspace(0, .4, int(128*(1-negthres))))
    colors1a = plt.cm.YlGnBu_r(np.linspace(0, 0.4, int(128 * (1 - negthres))))
    colors1b = plt.cm.Blues(np.linspace(1, 1, int(128 * negthres)))
    colors1b[:, -1] = 0
    colors1 = np.vstack((colors1b, colors1a))

    colors2a = plt.cm.gist_heat(np.linspace(0.8, 0, int(128 * (1 - posthres))))
    colors2b = plt.cm.gist_heat(np.linspace(0, 0, int(128 * posthres)))
    colors2b[:, -1] = 0
    colors2 = np.vstack((colors2a, colors2b))

    colors1 = colors1[::-1, :]
    colors2 = colors2[::-1, :]

    # make lower values slightly transparent
    #     colors1[96:128,-1]=np.linspace(1, 0, 32)
    #     colors2[0:32,-1]=np.linspace(0, 1, 32)

    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    if reverse:
        colors = colors[::-1, :]
    return mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)


# TODO: refactor using nilearn affine transforms
def coords(inputimage, i, j, k):

    """Returns X, Y, Z coordinates for i, j, k


    Args:
        inputimage(Niimg-like object): Image to get coordinates from
        i(int): i
        j(int): j
        k(int): k

    Returns:
        numpy.ndarray: X, Y, Z coordinates
    """
    M = inputimage.affine[:3, :3]
    abc = inputimage.affine[:3, 3]
    return M.dot([i, j, k]) + abc


def calculate_network_damage(lesion_mask, network_map, calculation, sign_thr="both"):
    """Calculates the network damage score for a given lesion

    Args:
        lesion_mask(niimg-like): Mask of the lesion
        network_mask(niimg-like): Network map of interest
        calculation(function): A function used to calculate the damage score
        sign_thr(str): Specifies which sign values to consider. Can either be 'pos', 'neg', or 'both'.
    """
    lesion_data = lesion_mask.get_fdata()
    net_data = network_map.get_fdata()
    if lesion_data.min() < 0:
        print("Warning: negative weights")
    masked_data = np.multiply(lesion_data, net_data)
    if sign_thr == "pos":
        thr_mask = ma.masked_where(masked_data < 0, masked_data)

    elif sign_thr == "neg":
        thr_mask = ma.masked_where(masked_data > 0, masked_data)
    else:
        thr_mask = masked_data
    # masked_img = image.new_img_like(network_map, masked_data)
    return calculation(thr_mask)


def fast_glass_brain_plot(
    map_to_plot,
    title="",
    output_file="./tmp_img/fast_img.png",
    colorbar=True,
    display_mode="lyrz",
    plot_abs=True,
    cmap=hotBlues(0, 0),
):
    """fast_glass_brain_plot creates a neuroimaging plot using plot_glass_brain.

    It first saves the file then load it back
    locally instead. It takes the same input as plot_glass_brain and the default settings are tuned from the previous
    analysis notebook.

    Args:
        map_to_plot(Niimg-like object): any data set up for nilearn plotting function
        output_file(str): where the plot is sent before being recalled. If no file is provided, the default is sent to a file
            called fast image in tmp_img. The function will create a /tmp_img/ directory if it does not currently exit.
        colorbar(bool, optional): Specifies if the colorbar should be displayed. Defaults to True.
        display_mode(str, optional): Chooses direction of cuts. Defaults to 'lyrz'
        plot_abs(bool, optional): Specifies if magnitude only is plotted. Defaults to True.
        cmap(func): cmap function to use. The function uses hotBlues as the default cmap and does not currently support another option.
    """

    # Improvement on this function could send it to cache directly.
    # The function provides the l y r z displays for the brain data:

    random_seed = str(random())[2:]
    if not os.path.exists("./tmp_img_" + random_seed + "/"):
        os.makedirs("./tmp_img_" + random_seed + "/")
    output_file = "./tmp_img_" + random_seed + "/fast_img.png"

    fig = plt.figure(figsize=(10, 20))
    plotting.plot_glass_brain(
        map_to_plot,
        plot_abs=plot_abs,
        cmap=cmap,
        colorbar=colorbar,
        display_mode=display_mode,
        title=title,
        output_file=output_file,
    )
    fast_img = mpimg.imread(output_file)
    shutil.rmtree("./tmp_img_" + random_seed + "/")
    plt.axis("off")
    plt.imshow(fast_img)


def fast_plot_brain_z(args):
    # fast_plot_brain_z saves the result of plotting_plot_stat_map in a local directory called /tmp_img/ after being passed
    # arguments from its main function, i.e fidl_plot2. It is supposed to be a worker function for the fidl_plot2
    # multiprocessing part

    pi, img_loc, map_to_plot, cmap_to_use, random_seed = args
    ax = list(range(0, 2))
    ax[1] = plt.subplot(111)
    last_slice = pi == 29
    fig = plt.figure(figsize=(16, 16))
    file_name = (
        "./tmp_img_" + random_seed + "/" + random_seed + "_img" + str(pi) + ".png"
    )
    plotting.plot_stat_map(
        map_to_plot,
        display_mode="z",
        cmap=cmap_to_use,
        colorbar=last_slice,
        cut_coords=[img_loc],
        output_file=file_name,
        axes=None,
    )  # plt.axes())


def fidl_plot2(
    map_to_plot,
    title_to_use="",
    cmap_to_use="",
    cmap="",
    filename_to_use="",
    posthres=0,
    negthres=0,
    posmax="",
    negmax="",
):
    """fidl_plot creates a grid of sliced brains similarly to fidl_plot but includes multiprocessing for faster processing.

    The function uses a worker function: fast_plot_brain_z, to create all its subplots in parallel and in different
    processes. The subplots are saved as files by the worker function. Once all the subprocesses have been performed, the
    main function calls a grid and then allocate files saved by the worker functionfrom the tmp_img folder.
    For easy retrieval, the function generate a random seed each time to be passed to its worker function.

    Args:
        map_to_plot(Niimg-like object): any nilearn plotting data
        cmap_to_use(func): pass cmap to use, default to hotBlues if none is given.
        filename_to_use(str): filename to save the final plot as. It does not interfere with the temporary files.
        posthres(int, optional): Positive threshold to pass to cmap. Defaults 0.
        negthres(int, optional): Negative threshold to pass to cmap. Defaults 0.
        posmax(str): ???
        negmax(str): ???
    """

    if cmap:
        cmap_to_use = cmap

    long_img = np.nan_to_num(image.load_img(map_to_plot).get_data())
    imgmax = np.max(long_img)
    imgmin = np.min(long_img)

    if posmax:
        if negmax:
            map_to_plot = image.math_img(
                "np.clip(img," + str(negmax) + "," + str(posmax) + ")", img=map_to_plot
            )
        else:
            map_to_plot = image.math_img(
                "np.clip(img," + str(imgmin) + "," + str(posmax) + ")", img=map_to_plot
            )
    else:
        if negmax:
            map_to_plot = image.math_img(
                "np.clip(img," + str(negmax) + "," + str(imgmax) + ")", img=map_to_plot
            )
        else:
            map_to_plot

    random_seed = str(random())[2:]
    if not os.path.exists("./tmp_img_" + random_seed + "/"):
        os.makedirs("./tmp_img_" + random_seed + "/")

    step = 4
    slices = list(range(-40, 74, step))[::-1]
    ax = []
    # fig, ax[0:31] = plt.subplots(4,8,figsize=(54,30))
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(4, 8)
    num_figs = 28

    if cmap_to_use:
        cmap_to_use
    else:
        cmap_to_use = hotBlues(posthres, negthres)

    pool = multiprocessing.Pool(processes=24)
    input = zip(
        list(range(1, 30, 1)),
        slices,
        [map_to_plot] * 29,
        [cmap_to_use] * 29,
        [random_seed] * 29,
    )
    pool.map(fast_plot_brain_z, input)

    ax = list(range(0, 30))
    for pi in list(range(1, 30)):
        #    #ax = fig.add_subplot(gs[pi])
        ax[pi] = plt.subplot(4, 8, pi)
        plt.axis("off")
        last_slice = pi == len(slices) - 1
        img = mpimg.imread(
            "./tmp_img_" + random_seed + "/" + random_seed + "_img" + str(pi) + ".png"
        )
        ax[pi].imshow(img)
        # imgplot = plt.imshow(img)
        # ax[pi].plot(imgPlot)
        # plt.imshow(img)
    plt.show()

    # Grid of subplots
    gs.update(wspace=0.05, hspace=0.05)
    ax = plt.subplot2grid((4, 7), (3, 5), colspan=3)
    ax.text(
        0.5,
        0.5,
        title_to_use,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")

    pool.terminate()
    pool.join()

    if filename_to_use:
        fig.savefig(filename_to_use)
    shutil.rmtree("./tmp_img_" + random_seed + "/")

    plt.show()


def longest_common_suffix(list_of_strings):
    """Return the longest common suffix in a list of strings. Helpful for
    separating out subject identifiers from other file data.

    Args:
        list_of_strings (list of str): Strings to extract common suffixes from

    Returns:
        suffix (str): Longest common suffix.
    """
    reversed_strings = [s[::-1] for s in list_of_strings]
    reversed_lcs = os.path.commonprefix(reversed_strings)
    lcs = reversed_lcs[::-1]
    return lcs


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def update_mongo_dataset(
    dataset_path, server_addr="circuittherapeutics-xnat.dipr.partners.org"
):
    """Update the mongodb server with metadata derived from bids json files

    Args:
        dataset_path (str): Path to dataset
        server_addr (str, optional): Address of MongoDB server. Defaults to 'circuittherapeutics-xnat.dipr.partners.org'.
    """
    dataset_desc = dataset_path + "/dataset_description.json"
    subject_dirs = glob.glob(dataset_path + "/sub-*")
    dataset_record = {}
    client = MongoClient(server_addr, 27017)
    db = client["dl_archive"]
    # Read dataset description
    with open(dataset_desc) as f:
        dataset_desc_data = json.load(f)
    dataset_record.update(dataset_desc_data)
    subject_records = []
    configfile = nimds.get_filepath("connectivity_config")
    layout = BIDSLayout(dataset_path, config=configfile)
    file_patterns = [
        "*.nii.gz",
        "*.nii",
        "*.gii",
        "*.gii.gz",
        "*.mat",
        "*.trk.gz",
        "*.connectivity.mat",
        "*.txt",
        "*.connectogram.txt",
        "*.node",
        "*.edge",
        "*.trk.gz",
    ]
    for sub in subject_dirs:
        sub_record = {}
        sub_record["subject"] = sub.split("sub-")[-1]
        files = []
        for p in file_patterns:
            files += find(p, sub)
        file_records = []
        # remove duplicates
        files = set(files)
        for f in files:
            file_record = {}
            name_data = layout.parse_file_entities(f)
            file_record.update(name_data)
            sidecar = ".".join([f.split(".")[0], "json"])
            file_record["path"] = f
            # Remove redundant fields
            file_record.pop("subject", None)
            file_record.pop("extension", None)
            file_records.append(file_record)
        sub_record["files"] = file_records
        subject_records.append(sub_record)
    dataset_record["subjects"] = subject_records
    if server_addr == "DEBUG":
        print(dataset_record)
        return
    result = db.archive.update_one(
        {"Name": dataset_record["Name"]}, {"$set": dataset_record}, upsert=True
    )

    print(result)


# TODO: Consider deprecation
def mgi_autoplot_box(
    ROI_df,
    n_ROIs,
    short_ROIs_Names,
    short_dataset_names,
    plot,
    title,
    posthoc_ttest=False,
):
    if n_ROIs == 1:
        fig = plt.figure(figsize=(12, 12))
        ax = list(range(1, 3))
        ax[1] = plt.subplot(111)
        ROI_df[0].plot(
            kind=plot,
            ax=ax[1],
            title=short_ROIs_Names[0],
            fontsize=15,
            showfliers=False,
        )
        ax[1].title.set_size(30)
        fig.suptitle(title, fontsize=40)
        axtick = ax[1].get_xticks()
        for sdni in list(range(1, len(short_dataset_names))):
            name = short_dataset_names[sdni]
            t_stat, p_value = stats.ttest_ind(
                ROI_df[0][short_dataset_names[0]],
                ROI_df[0][name],
                equal_var=False,
                nan_policy="omit",
            )
            textstr = "T=" + "{:.4f}".format(t_stat) + "\np=" + "{:.6f}".format(p_value)
            ax[1].text(
                axtick[sdni],
                -0.4,
                textstr,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=12,
                ha="center",
            )

    else:
        fig = plt.figure(figsize=(24, 6 * n_ROIs))
        ax = list(range(1, n_ROIs + 2))

        if n_ROIs % 2 != 0:
            iter_ROIs = n_ROIs
            n_ROIs = n_ROIs + 1
        else:
            iter_ROIs = n_ROIs

        for i in list(range(1, iter_ROIs + 1)):
            ax[i] = plt.subplot(1, n_ROIs, i)
            ROI_df[i - 1].plot(
                kind=plot,
                ax=ax[i],
                title=short_ROIs_Names[i - 1],
                fontsize=15,
                showfliers=False,
            )
            ax[i].title.set_size(30)
            fig.suptitle(title, fontsize=40)
            axtick = ax[i].get_xticks()
            aytick = ax[i].get_yticks()
            ticks_labels = [short_dataset_names[0]]
            for sdni in list(range(1, len(short_dataset_names))):
                name = short_dataset_names[sdni]
                t_stat, p_value = stats.ttest_ind(
                    ROI_df[i - 1][short_dataset_names[0]],
                    ROI_df[i - 1][name],
                    equal_var=False,
                    nan_policy="omit",
                )
                textstr = (
                    short_dataset_names[sdni]
                    + "\nT="
                    + "{:.4f}".format(t_stat)
                    + "\np="
                    + "{:.6f}".format(p_value)
                )
                ticks_labels.append(textstr)
            ax[i].set_xticklabels(ticks_labels)

        if posthoc_ttest:
            print("Post hoc two sample t-test:")
            all_tests_df = pd.DataFrame()
            for sdni1 in list(range(0, len(short_dataset_names))):
                list_out = []
                for sdni2 in list(range(0, len(short_dataset_names))):
                    name = short_dataset_names[sdni]
                    t_stat, p_value = stats.ttest_ind(
                        ROI_df[0][short_dataset_names[sdni1]],
                        ROI_df[0][short_dataset_names[sdni2]],
                        equal_var=False,
                        nan_policy="omit",
                    )
                    list_out.append(
                        str(round(t_stat, 5)) + " (" + str(round(p_value, 5)) + ")"
                    )
                all_tests_df[short_dataset_names[sdni1]] = list_out
            all_tests_df.index = short_dataset_names
            all_tests_df
            return all_tests_df


# TODO: Consider deprecation
def mgi_autoplot_bar(
    RList,
    n_ROIs,
    short_ROIs_Names,
    mean,
    error,
    title,
    opacity=0.5,
    error_config={"ecolor": "0.3"},
    error_bar_size=0,
    edges="white",
):

    if n_ROIs == 1:
        fig = plt.figure(figsize=(12, 12))
        ax = list(range(1, 3))
        for i in list(range(1, n_ROIs + 1)):
            ax[1] = plt.subplot(111)
            RList[i - 1][mean].plot.bar(
                ax=ax[1],
                yerr=RList[i - 1][error],
                title=short_ROIs_Names[0],
                fontsize=15,
                rot=0,
                alpha=opacity,
                error_kw=error_config,
                edgecolor=edges,
                capsize=error_bar_size,
                width=1,
            )
            ax[i].title.set_size(30)
            ax[i].set_xlabel("")
            fig.suptitle(title, fontsize=40)

    else:
        fig = plt.figure(figsize=(24, 6 * n_ROIs))
        ax = list(range(1, n_ROIs + 2))
        for i in list(range(1, n_ROIs + 1)):
            ax[i] = plt.subplot(1, n_ROIs, i)
            RList[i - 1][mean].plot.bar(
                ax=ax[i],
                yerr=RList[i - 1][error],
                title=short_ROIs_Names[i - 1],
                fontsize=15,
                rot=0,
                alpha=opacity,
                error_kw=error_config,
                edgecolor=edges,
                capsize=error_bar_size,
                width=1,
            )
            ax[i].title.set_size(30)
            ax[i].set_xlabel("")
            fig.suptitle(title, fontsize=40)


def spatial_corr(img1, img2, mask=BRAIN_MASK):
    """Calculate spatial correlation between two images.

    Args:
        img1 (Niimg-like): First img
        img2 (Niimg-like): Second img
        mask (Niimg-like): binary mask img. Defaults to BRAIN_MASK (MNI152_T1_2mm_brain_mask_dil)

    Returns:
        float: Pearson's r correlation.
    """
    masker = maskers.NiftiMasker(mask, standardize=False).fit()
    vec1 = masker.transform(img1)[0, :]
    vec2 = masker.transform(img2)[0, :]
    return sp.stats.pearsonr(vec1, vec2)[0]
