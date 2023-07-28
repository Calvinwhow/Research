#!/bin/python3
# Volume connectivity methods referenced Andreas Horn's connectome.sh script from LeadDBS
# Author: Christopher Lin <clin5@bidmc.harvard.edu> (Volume Connectivity)
# Author: William Drew <wdrew@bwh.harvard.edu> (Surface Connectivity)


import os
import csv
import time
import argparse
import numpy as np
import nibabel as nib

from nimlab import connectomics as cs
from nimlab import datasets as nimds
from nimlab import functions as nimfs
from nimlab import surface as nimsf
from collections import OrderedDict
from nilearn import image, maskers
from natsort import natsorted
from glob import glob
from tqdm import tqdm


# Volume Masks
VOLUME_MASK_SIZES = {
    "222": 285903,
    "mni_icbm152": 225222,
    "MNI152_T1_2mm_brain_mask": 228483,
    "MNI152_T1_2mm_brain_mask_dil1": 262245,
    "MNI152_T1_2mm_brain_mask_dil": 292019,
}

# Surface Masks
SURFACE_MASK_SIZES = {"fsaverage5": 18426}
SURFACE_MASK_SPLITS = {"fsaverage5": 9204}


def connectome_type(connectome_file):
    # Detects whether a normative connectome file is in Volume-space or Surface-space
    #  - Currently a hacky detection method that uses the shape of the .npy matrix,
    #    will be swapped to connectome .json's (WD July 20, 2020)
    #  - WARNING: This function only checks 1 file's shape
    #
    # Author: William Drew (wdrew@bwh.harvard.edu)
    #
    # Arguments:
    #  - connectome_file (str): Path to a connectome/timecourse file
    #
    # Outputs:
    #  - conn_type (str): "volume" (Volume-space connectome)
    #                                     OR
    #                     "surface" (Surface-space connectome)

    conn_type = None

    # Check the shape of the connectome file passed
    connectome_file = np.load(connectome_file)

    if connectome_file.shape[1] in SURFACE_MASK_SIZES.values():
        conn_type = "surface"
    elif connectome_file.shape[1] in VOLUME_MASK_SIZES.values():
        conn_type = "volume"
    else:
        raise ValueError(
            f"Connectome Space Unrecognized: {'/'.join(connectome_file.split('/')[:-1])}"
        )

    return conn_type


def roi_type(roi_files):
    # Detects whether a list of ROIs is in Volume-space or Surface-space
    # If a file has extension .nii.gz, it is assumed to be volume-space.
    # If a file has extension .gii, it is assumed to be surface-space.
    #
    # Author: William Drew (wdrew@bwh.harvard.edu)
    #
    # Arguments:
    #  - roi_files (list): list of lists containing file paths to ROI masks
    #
    # Outputs:
    #  - roi_type (str): "volume" (Volume-space connectome) or "surface" (Surface-space connectome)
    roi_type = None
    roi_paths = np.array(roi_files)

    # If given a single column of files, check to see if they are in MNI152_2mm volume space
    #  - (check if dimensions are (91,109,91))
    if roi_paths.shape[1] == 1:
        for path in roi_paths:
            # Check to see if image has correct shape (for MNI152 2mm volume space)
            if image.get_data(path[0]).shape != (91, 109, 91):
                raise ValueError(
                    f"ROI Image {path} has invalid dimensions. If using Volume-space ROIs, please provide a one-column CSV of paths to .nii/.nii.gz Nifti ROI images. Only MNI152 2mm space (91x109x91) is supported at this time."
                )
        roi_type = "volume"

    # Else if given a double column of files, check to see if they are in Gifti fsaverage5 format
    #  - (check if dimensions are (10242,))
    elif roi_paths.shape[1] == 2:
        roi_paths = roi_paths.flatten()
        for path in roi_paths:
            # Check to see if a hemisphere file is missing
            if ".gii" not in path:
                raise ValueError(
                    "Invalid CSV of ROI Images. When using surface-space ROIS, ensure that you provide a full two-column CSV of paths to .gii Gifti ROI images where each row corresponds to a subject, the first column corresponds to the subject's Left Hemisphere ROI mask, and the second column corresponds to the subject's Right Hemisphere ROI mask."
                )
            # Check to see if image has correct shape (for fsaverage5 surface space)
            if nib.load(path).agg_data().shape != (10242,):
                raise ValueError(
                    f"ROI Image {path} has invalid dimensions. If using Surface-space ROIs, please provide a two-column CSV of paths to .gii Gifti ROI images. Only fsaverage5 space (10242,) is supported at this time."
                )
        roi_type = "surface"

    # if the provided CSV does not have either one or two columns, raise ValueError
    else:
        raise ValueError(
            "Please provide either a one-column CSV of paths to volume space Nifti files or a two-column CSV of paths to surface space Pseudo Gifti files"
        )

    return roi_type


def get_subject_name(subject_path, mode):
    # gets subject names by stripping things like path, '.nii'/'.gii'
    # from filename
    if mode == "volume":
        return os.path.basename(subject_path).split(".nii")[0]
    elif mode == "surface":
        lh_fname = os.path.basename(subject_path.split("|")[0])
        rh_fname = os.path.basename(subject_path.split("|")[1])
        # Sketchy method to detect the two surface naming formats:

        # Format 1: If filenames start with "lh." or "rh."
        # Get subject name by stripping leading "lh." and trailing ".gii" from left hemisphere filename
        if lh_fname[:3] == "lh." and rh_fname[:3] == "rh.":
            return lh_fname.split("lh.")[1].split(".gii")[0]
        # Format 2: If filenames don't start with "lh." and "rh.", assume it's in BIDS format
        # Get subject name by removing "_hemi-L" then stripping trailing ".gii" from the left hemisphere filename.
        else:
            return lh_fname.replace("_hemi-L", "").split(".gii")[0]
    else:
        raise ValueError(f"Invalid mode {mode}")


def get_leftRight_names(subject_path):
    # Returns a two element list of paths with 'lh.' and 'rh.' prepended to the input path
    return ["lh." + subject_path, "rh." + subject_path]


def make_subjects(
    connectome_file_tuples,
    roi_list,
    masker_list,
    command,
    roi_connectome_type,
    brain_connectome_type,
    same_connectome,
    warning_flag,
):
    # Creates the ConnectomeSubject objects that are used in the nimlab.connectomics module
    subjects = []
    if command == "seed":
        roi_dict = {}
    elif command == "matrix":
        roi_dict = OrderedDict()
    else:
        raise ValueError("Unrecognized command")
    if roi_connectome_type == "volume":
        for rois in roi_list:
            roi_masked = masker_list[0][0].transform(rois[0])
            roi_dict.update([(rois[0], roi_masked)])

    elif roi_connectome_type == "surface":
        for rois in roi_list:
            roi_masked = np.concatenate(
                [
                    np.reshape(masker_list[0][0].transform(rois[0]), (1, -1)),
                    np.reshape(masker_list[0][1].transform(rois[1]), (1, -1)),
                ],
                axis=1,
            )
            roi_dict.update([(rois[0] + "|" + rois[1], roi_masked)])

    for f in connectome_file_tuples:
        # f[0] is roi_connectome tuple, f[1] is brain_connectome tuple
        sub = cs.ConnectomeSubject(
            f[0],
            f[1],
            roi_dict,
            roi_connectome_type,
            brain_connectome_type,
            same_connectome,
            warning_flag,
        )
        subjects.append(sub)
    return subjects


if __name__ == "__main__":
    start = time.time()
    # Get arguments
    parser = argparse.ArgumentParser(
        description="Compute connectivity maps. Based off of LeadDBS's cs_fmri_conseed"
    )
    parser.add_argument(
        "-cs",
        metavar="roi_connectome",
        help="Folder containing connectome/timecourse files for ROIs",
        required=True,
    )
    parser.add_argument(
        "-csbrain",
        metavar="brain_connectome",
        help="Folder containing connectome/timecourse files for Brain. If not set, uses same connectome as -cs.",
    )
    parser.add_argument("-r", metavar="ROI", help="CSV of ROI images", required=True)
    """Argument: -r [CSV of ROI images]
    
      - If using volume-space ROIs, provide a one-column CSV of paths to .nii.gz ROI
        images where each row corresponds to a subject's ROI mask in volume-space.
      - <roi> will be used as the subject name for naming output files.
    
        Format:
            /path/to/subject1/roi1.nii.gz
            /path/to/subject2/roi2.nii.gz
            /path/to/subject3/roi3.nii.gz
                         .
                         .
                         .
    
      - If using surface-space ROIs, provide a two-column CSV of paths to .gii
        Gifti ROI images where each row corresponds to a subject, the first
        column corresponds to the subject's Left Hemisphere ROI mask, and the second
        column corresponds to the subject's Right Hemisphere ROI mask.
      - For correct naming of output files, two filename formats are accepted.
    
        Format (1): Filenames must begin with 'lh.' or 'rh.', followed by subject name, 
        followed by the file extension '.gii'.
            
            /path/to/subject1/lh.roi1.gii,/path/to/subject1/rh.roi1.gii
            /path/to/subject2/lh.roi2.gii,/path/to/subject2/rh.roi2.gii
            /path/to/subject3/lh.roi3.gii,/path/to/subject3/rh.roi3.gii
                                        .
                                        .
                                        .

        Format (2): Filenames must be in BIDS format, with hemisphere specified with the '_hemi-<L|R>' entity.

            /path/to/subject1/sub-roi1_hemi-L_atrophyMap.gii,/path/to/subject1/sub-roi1_hemi-R_atrophyMap.gii
            /path/to/subject2/sub-roi2_hemi-L_atrophyMap.gii,/path/to/subject2/sub-roi2_hemi-R_atrophyMap.gii
            /path/to/subject3/sub-roi3_hemi-L_atrophyMap.gii,/path/to/subject3/sub-roi3_hemi-R_atrophyMap.gii
                                                            .
                                                            .
                                                            .
    """

    parser.add_argument(
        "-bs",
        metavar="roi_mask_space",
        help="Binary image that specifies which voxels \
    of the brain to include in compressed ROI timecourse files. If ROI Connectome is in Volume-space, \
    defaults to the MNI152_T1_2mm_brain_mask_dil.nii.gz image included with FSL. If ROI Connectome is in\
    Surface-space, defaults to the fs5_mask_lh.gii and fs5_mask_rh.gii images created by William Drew.\
    NOTE: The roi_mask_space must FIRST be used to generate the ROI connectome files.",
    )
    parser.add_argument(
        "-bsbrain",
        metavar="brain_mask_space",
        help="Binary image that specifies which voxels\
    of the brain to include in compressed Brain timecourse files. If Brain Connectome is in Volume-space,\
    defaults to the MNI152_T1_2mm_brain_mask_dil.nii.gz image included with FSL. If Brain Connectome is in\
    Surface-space, defaults to the fs5_mask_lh.gii and fs5_mask_rh.gii images created by William Drew.\
    NOTE: The brain_mask_space must FIRST be used to generate the brain connectome files.",
    )

    parser.add_argument(
        "-mout", metavar="output_mask", help="Output mask. Defaults to no masking."
    )
    parser.add_argument(
        "-c",
        metavar="command",
        help="Seed or ROI matrix. Defaults to seed.",
        default="seed",
    )
    parser.add_argument(
        "-o", metavar="output", help="Output directory", default="seed", required=True
    )
    parser.add_argument(
        "-w",
        metavar="workers",
        help="Number of workers. Defaults to 12",
        type=int,
        default=int(12),
    )
    parser.add_argument(
        "--showwarnings",
        help="Activate flag to suppress warnings",
        default=True,
        action="store_false",
    )  # Defaulting to True is kinda weird, but it makes more sense to have a rarely used flag disable warnings rather than having to pass booleans
    parser.add_argument(
        "-fout",
        metavar="file_output",
        help="Output individual Fz files for each connectome subject. This can potentially generate vast quantities of data. USE WITH CARE!",
        type=str,
        default="",
    )
    args = parser.parse_args()

    # Process arguments
    output_folder = os.path.abspath(args.o)
    roi_file_list = os.path.abspath(args.r)
    cs_roi = os.path.abspath(args.cs)

    # Check if output directory exists
    if not os.path.exists(output_folder):
        raise ValueError("Invalid output folder: " + output_folder)

    # If brain connectome is not set, set to be the same as ROI connectome
    if args.csbrain is None:
        cs_brain = cs_roi
    else:
        cs_brain = os.path.abspath(args.csbrain)

    # Load paths to all ROI connectome files
    roi_connectome_files_norms = natsorted(glob(cs_roi + "/*_norms.npy"))
    roi_connectome_files = [
        (glob(f.split("_norms")[0] + ".npy")[0], f) for f in roi_connectome_files_norms
    ]
    if len(roi_connectome_files) == 0:
        raise ValueError("No roi connectome files found")
    if len(roi_connectome_files) == 1:
        print("Single Connectome File detected. Only Fz files will be created.")

    # Load paths to all Brain connectome files
    brain_connectome_files_norms = natsorted(glob(cs_brain + "/*_norms.npy"))
    brain_connectome_files = [
        (glob(f.split("_norms")[0] + ".npy")[0], f)
        for f in brain_connectome_files_norms
    ]
    if len(brain_connectome_files) == 0:
        raise ValueError("No brain connectome files found")

    connectome_files = list(zip(roi_connectome_files, brain_connectome_files))

    # Check ROI and Brain connectome types using the first connectome file in each list
    roi_connectome_type = connectome_type(roi_connectome_files[0][0])
    brain_connectome_type = connectome_type(brain_connectome_files[0][0])

    if roi_connectome_type != brain_connectome_type:
        for conn_subject in connectome_files:
            ((roi_tc, roi_norm), (brain_tc, brain_norm)) = conn_subject
            if os.path.basename(roi_tc) != os.path.basename(
                brain_tc
            ) or os.path.basename(roi_norm) != os.path.basename(brain_norm):
                raise ValueError(
                    f"The selected ROI Connectome and Brain Connectome are incompatible with each other. \
                Since you are using an ROI {roi_connectome_type}-space connectome with a Brain {brain_connectome_type}-space connectome\
                (different space types), the datasets used to make the connectomes must be identical with identical subject naming.\
                \nFor example, the GSP1000_MF (volume) and GSP1000_MF_surf_fs5 (surface) connectomes were generated from the same \
                subjects with identical connectome subject naming."
                )

    # Get ROI Connectome Mask
    if args.bs is None:
        if roi_connectome_type == "volume":
            roi_mask_img = [nimds.get_img("MNI152_T1_2mm_brain_mask_dil")]
        elif roi_connectome_type == "surface":
            roi_mask_img = [nimds.get_img("fs5_mask_lh"), nimds.get_img("fs5_mask_rh")]
    else:
        if roi_connectome_type == "volume":
            roi_mask_img = [image.load_img(args.bs)]
        elif roi_connectome_type == "surface":
            if args.bs == "fsaverage5":
                brain_mask_img = [
                    nimds.get_img("fs5_mask_lh"),
                    nimds.get_img("fs5_mask_rh"),
                ]
            else:
                raise ValueError(
                    "Only the fsaverage5 mask is supported. If you wish to use a surface connectome with your ROIs, omit the -bs flag"
                )

    # Get Brain Connectome Mask
    if args.bsbrain is None:
        if brain_connectome_type == "volume":
            brain_mask_img = [nimds.get_img("MNI152_T1_2mm_brain_mask_dil")]
        elif brain_connectome_type == "surface":
            brain_mask_img = [
                nimds.get_img("fs5_mask_lh"),
                nimds.get_img("fs5_mask_rh"),
            ]
    else:
        if brain_connectome_type == "volume":
            brain_mask_img = [image.load_img(args.bsbrain)]
        elif brain_connectome_type == "surface":
            if args.bsbrain == "fsaverage5":
                brain_mask_img = [
                    nimds.get_img("fs5_mask_lh"),
                    nimds.get_img("fs5_mask_rh"),
                ]
            else:
                raise ValueError(
                    "Only the fsaverage5 mask is supported. If you wish to use a surface connectome for the brain, omit the -bsbrain flag"
                )

    # Generate mask lists
    if roi_connectome_type == "volume":
        roi_masker_list = [maskers.NiftiMasker(mask).fit() for mask in roi_mask_img]
    elif roi_connectome_type == "surface":
        roi_masker_list = [nimsf.GiftiMasker(mask) for mask in roi_mask_img]
    if brain_connectome_type == "volume":
        brain_masker_list = [maskers.NiftiMasker(mask).fit() for mask in brain_mask_img]
    elif brain_connectome_type == "surface":
        brain_masker_list = [nimsf.GiftiMasker(mask) for mask in brain_mask_img]
    masker_list = [roi_masker_list, brain_masker_list]

    # Generate List of lists of ROI Files
    # If Volume ROIs: [[roi1.nii.gz],[roi2.nii.gz],...]
    # If Surface ROIs: [[lh.roi1.gii, rh.roi1.gii],[lh.roi2.gii, rh.roi2.gii],...]
    roi_files = []
    flist = open(roi_file_list)
    reader = csv.reader(flist, delimiter=",")
    for f in reader:
        roi_files.append(f)
    flist.close()

    # Check that ROI Connectome is same type as ROI files
    if roi_connectome_type != roi_type(roi_files):
        raise ValueError(
            f"ROI Space type ({roi_type(roi_files)}) is not the same as ROI Connectome Space type ({roi_connectome_type})!"
        )

    # If the ROI connectome and the Brain connectome are the same connectome (not just the same type, the same connectome), set a flag
    # that allows us to elect not to double load the ROI and Brain connectomes in connectomics.make_fz_maps()
    same_connectome = cs_brain == cs_roi

    subjects = make_subjects(
        connectome_files,
        roi_files,
        masker_list,
        args.c,
        roi_connectome_type,
        brain_connectome_type,
        same_connectome,
        args.showwarnings,
    )

    print(f"Loaded {str(len(roi_files))} ROIs")
    print(f"Using {str(len(roi_connectome_files))} ROI connectome files")
    print(f"Using {str(len(brain_connectome_files))} Brain connectome files")
    print(f"Using ROI Space: {roi_connectome_type}")
    print(f"Using Brain Space: {brain_connectome_type}")

    if brain_connectome_type == "volume":
        map_kinds = ["_AvgR_Fz.nii.gz", "_AvgR.nii.gz", "_T.nii.gz"]
    if brain_connectome_type == "surface":
        map_kinds = ["_AvgR_Fz.gii", "_AvgR.gii", "_T.gii"]

    if args.c == "seed":
        print("Computing maps")
        if len(subjects) > 1:
            avgR_fz_maps, avgR_maps, T_maps = cs.calculate_maps(
                subjects,
                args.w,
                output_folder,
                args.fout,
                single_connectome_subject=False,
            )
            if brain_connectome_type == "volume":
                for keys in zip(avgR_fz_maps.keys(), avgR_maps.keys(), T_maps.keys()):
                    # Do some voodoo majik and get a list of properly formatted output file names ~WD
                    fnames = [
                        get_subject_name(key, roi_connectome_type) + map_type
                        for (key, map_type) in zip(keys, map_kinds)
                    ]
                    masker_list[1][0].inverse_transform(
                        avgR_fz_maps[keys[0]]
                    ).to_filename(os.path.join(output_folder, fnames[0]))
                    masker_list[1][0].inverse_transform(avgR_maps[keys[1]]).to_filename(
                        os.path.join(output_folder, fnames[1])
                    )
                    masker_list[1][0].inverse_transform(T_maps[keys[2]]).to_filename(
                        os.path.join(output_folder, fnames[2])
                    )

            elif brain_connectome_type == "surface":
                for keys in zip(avgR_fz_maps.keys(), avgR_maps.keys(), T_maps.keys()):
                    # Do some voodoo majik and get a list of properly formatted output file names ~WD
                    fnames = [
                        get_leftRight_names(
                            get_subject_name(key, roi_connectome_type) + map_type
                        )
                        for (key, map_type) in zip(keys, map_kinds)
                    ]
                    masker_list[1][0].inverse_transform(
                        avgR_fz_maps[keys[0]][: SURFACE_MASK_SPLITS["fsaverage5"]]
                    ).to_filename(os.path.join(output_folder, fnames[0][0]))
                    masker_list[1][1].inverse_transform(
                        avgR_fz_maps[keys[0]][SURFACE_MASK_SPLITS["fsaverage5"] :]
                    ).to_filename(os.path.join(output_folder, fnames[0][1]))
                    masker_list[1][0].inverse_transform(
                        avgR_maps[keys[1]][: SURFACE_MASK_SPLITS["fsaverage5"]]
                    ).to_filename(os.path.join(output_folder, fnames[1][0]))
                    masker_list[1][1].inverse_transform(
                        avgR_maps[keys[1]][SURFACE_MASK_SPLITS["fsaverage5"] :]
                    ).to_filename(os.path.join(output_folder, fnames[1][1]))
                    masker_list[1][0].inverse_transform(
                        T_maps[keys[2]][: SURFACE_MASK_SPLITS["fsaverage5"]]
                    ).to_filename(os.path.join(output_folder, fnames[2][0]))
                    masker_list[1][1].inverse_transform(
                        T_maps[keys[2]][SURFACE_MASK_SPLITS["fsaverage5"] :]
                    ).to_filename(os.path.join(output_folder, fnames[2][1]))
        if len(subjects) == 1:
            cs.calculate_maps(
                subjects,
                args.w,
                output_folder,
                args.fout,
                single_connectome_subject=True,
            )
    elif args.c == "matrix":
        print("Computing matrices")
        avgR_fz_mat, avgR_mat, T_mat, roi_names = cs.calculate_roi_matrix(
            subjects, args.w
        )
        np.savetxt(
            output_folder + "/matrix_corrMx_AvgR_Fz.csv", avgR_fz_mat, delimiter=","
        )
        np.savetxt(output_folder + "/matrix_corrMx_AvgR.csv", avgR_mat, delimiter=",")
        np.savetxt(output_folder + "/matrix_corrMx_T.csv", T_mat, delimiter=",")
        name_file = open(output_folder + "/matrix_corrMx_names.csv", "w+")
        for n in roi_names:
            if brain_connectome_type == "surface":
                n = list(nimfs.lcs(n.split("|")[0], n.split("|")[1]))[0].split("_", 1)[
                    1
                ]
            name_file.write(n)
            name_file.write("\n")
        name_file.close()

    else:
        print("Unrecognized command")

    # Convert npy files for individual connectome output to niftis
    if args.fout:
        print("Transforming individual npy files")
        ind_files = glob(args.fout + "/*/*.npy")
        for f in tqdm(ind_files):
            ind_npy = np.load(f)
            if brain_connectome_type == "surface":
                ind_img_lh = masker_list[1][0].inverse_transform(
                    ind_npy[: SURFACE_MASK_SPLITS["fsaverage5"]]
                )
                ind_img_rh = masker_list[1][1].inverse_transform(
                    ind_npy[SURFACE_MASK_SPLITS["fsaverage5"] :]
                )
                sf_fname = os.path.basename(f).split(".npy")[0]
                sf_out_dir = os.path.dirname(f)
                ind_img_lh.to_filename(
                    os.path.join(sf_out_dir, "lh." + sf_fname + ".gii")
                )
                ind_img_rh.to_filename(
                    os.path.join(sf_out_dir, "rh." + sf_fname + ".gii")
                )
            elif brain_connectome_type == "volume":
                ind_img = masker_list[1][0].inverse_transform(ind_npy)
                ind_nifti_fname = f.split(".")[0]
                ind_img.to_filename(ind_nifti_fname + ".nii.gz")
            if os.path.exists(f):
                os.remove(f)
    # Convert npy files for individual connectome output to niftis
    if len(subjects) == 1:
        print("Transforming individual npy files")
        ind_files = glob(output_folder + "/*/*.npy")
        for f in tqdm(ind_files):
            ind_npy = np.load(f)
            if brain_connectome_type == "surface":
                ind_img_lh = masker_list[1][0].inverse_transform(
                    ind_npy[: SURFACE_MASK_SPLITS["fsaverage5"]]
                )
                ind_img_rh = masker_list[1][1].inverse_transform(
                    ind_npy[SURFACE_MASK_SPLITS["fsaverage5"] :]
                )
                sf_fname = os.path.basename(f).split(".npy")[0]
                sf_out_dir = os.path.dirname(f)
                ind_img_lh.to_filename(
                    os.path.join(sf_out_dir, "lh." + sf_fname + "_Fz.gii")
                )
                ind_img_rh.to_filename(
                    os.path.join(sf_out_dir, "rh." + sf_fname + "_Fz.gii")
                )
            elif brain_connectome_type == "volume":
                ind_img = masker_list[1][0].inverse_transform(ind_npy)
                ind_nifti_fname = f.split(".")[0]
                ind_img.to_filename(ind_nifti_fname + "_Fz.nii.gz")
            if os.path.exists(f):
                os.remove(f)
    end = time.time()
    elapsed = end - start
    print("Total elapsed: " + str(elapsed))
    print("Avg time per seed: " + str(elapsed / len(roi_files)))
