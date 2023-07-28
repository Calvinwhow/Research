import os
import sys
import time
import gzip
import json
import shutil
import hashlib
import pickle
import numpy as np
import pandas as pd
import nibabel as nib

from glob import glob
from tqdm import tqdm
from datetime import date
from bids import BIDSLayout
from termcolor import cprint
from scipy.io import savemat
from natsort import natsorted
from nilearn import plotting, image
from matplotlib import pyplot as plt

from nimlab import nimtrack
from nimlab import surface as nimsf
from nimlab import datasets as nimds
from nimlab import configuration as config
from nimlab.functions import sum_imgs, update_mongo_dataset

from IPython.core.getipython import get_ipython
from IPython.display import display

FSL_MNI152_2MM_AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

FSL_MNI152_1MM_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

FSAVERAGE_SHAPES = [(10242,)]


def hash_file(path):
    hasher = hashlib.md5()
    with open(path, "rb") as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()


def init_env(project_name, preprocess_folder):
    """Initialize environment dictionary. If env pickle exists, load and return it.

    Parameters
    ----------
    project_name : str
        Name of project.
    preprocess_folder : str
        Preprocessing folder where dataset is built. If empty string, dataset will be
        built at current working directory.

    Returns
    -------
    environment : dict
        Environment variable dictionary. Contains metadata and dataframes. If the
        notebook has been run before, an environment pickle should already exist - just
        load it. If the notebook hasn't been run before, a new environment dictionary
        will be generated and returned.

    """
    ipython = get_ipython()
    if preprocess_folder == "":
        preprocess_folder = os.getcwd()
    project_path = os.path.abspath(os.path.join(preprocess_folder, project_name))
    os.makedirs(project_path, exist_ok=True)
    cprint(f"The project directory is {project_path}", "green")
    env_pickle_path = os.path.join(project_path, f"{project_name}_environment.pkl")
    if os.path.exists(env_pickle_path):
        environment = pickle.load(open(env_pickle_path, "rb"))
        cprint(f"Environment variables loaded!", "green", attrs=["bold"])
    else:
        # Create datalad repository
        pickle_name = f"{project_name}_environment.pkl"
        ipython.system(f"datalad create -c text2git {project_path}")
        ipython.system(f"echo .ipynb_checkpoints > {project_path}/.gitignore")
        ipython.system(f"echo {pickle_name} >> {project_path}/.gitignore")
        ipython.system(f"datalad save -d {project_path} -m 'Added .gitignore'")
        environment = {
            "project_name": project_name,
            "preprocess_folder": preprocess_folder,
            "project_path": project_path,
            "env_pickle_path": env_pickle_path,
            "brain_mask_1mm": nimds.get_img("MNI152_T1_1mm_brain_mask"),
            "brain_mask_2mm": nimds.get_img("MNI152_T1_2mm_brain_mask"),
            "input_spaces": [],
            "output_spaces": [],
            "connectivity_analyses": [],
        }
        pickle.dump(environment, open(env_pickle_path, "wb"))
        cprint(f"Environment variable initialized!", "green", attrs=["bold"])
    return environment


def save_env(environment):
    pickle.dump(environment, open(environment["env_pickle_path"], "wb"))


def load_rois(input_path):
    """Load ROIs from a directory and sort into dicts

    Parameters
    ----------
    input_path : str or list of paths
        Path to directory containing .nii.gz, .nii, or .gii input ROI masks OR list of
        paths to .nii.gz, .nii, or .gii files

    Returns
    -------
    lesion_type : str
        Lesion space type ("volume" or "surface")
    lesions : dict
        Dictionary of subject name and file path to ROI file(s)

    """
    if type(input_path) is list:
        volume_lesion_files = natsorted([p for p in input_path if ".nii" in p])
        surface_lesion_files = natsorted([p for p in input_path if ".gii" in p])
    else:
        volume_lesion_files = natsorted(glob(os.path.join(input_path, "*.nii*")))
        surface_lesion_files = natsorted(glob(os.path.join(input_path, "*.gii")))

    if len(volume_lesion_files) > 0 and len(surface_lesion_files) > 0:
        raise ValueError(
            "You cannot use both volume-space and surface-space files as inputs. Consider using two separate notebooks, one for volume-space ROIs and one for surface-space ROIs."
        )

    lesions = {}
    if len(volume_lesion_files) > 0:
        lesion_type = "volume"
        for vol_file in volume_lesion_files:
            subject_name = (
                os.path.basename(vol_file).split(".nii")[0].replace("_lesionMask", "")
            )
            lesions[subject_name] = vol_file
        print("I found", len(volume_lesion_files), "volume-space lesion files:")
        for lesion in volume_lesion_files[0:5]:
            print(lesion)
    elif len(surface_lesion_files) > 0:
        lesion_type = "surface"
        for surf_file in surface_lesion_files:
            subject_name = (
                os.path.basename(surf_file)
                .split(".")[1:-1][0]
                .replace("_lesionMask", "")
            )  # strip leading lh./rh. and trailing .gii. Removes "_lesionMask" if present. Should probably be done with regex
            hemi = os.path.basename(surf_file).split(".")[0]
            extension = os.path.basename(surf_file).split(".")[-1]
            if hemi not in ["lh", "rh"]:
                raise ValueError(
                    f"ROI File {surf_file} does not have a correctly named hemisphere. Make sure 'lh.' or 'rh.' is at the beginning of your filename"
                )
            elif extension not in ["gii"]:
                raise ValueError(
                    f"ROI File {surf_file} does not have the correct extension for a surface file. Make sure the file has extension '.gii' and is not zipped (not .gii.gz)"
                )
            else:
                if subject_name in lesions:
                    lesions[subject_name][hemi] = surf_file
                else:
                    lesions[subject_name] = {hemi: surf_file}
        for subject in lesions:
            if "lh" not in lesions[subject]:
                raise ValueError(f"Subject {subject} is missing a left hemisphere file")
            elif "rh" not in lesions[subject]:
                raise ValueError(
                    f"Subject {subject} is missing a right hemisphere file"
                )
        print("I found", len(list(lesions.keys())), "surface-space lesion subjects:")
        for subject in list(lesions.keys())[:5]:
            print(subject)
    else:
        raise ValueError(f"No ROIs were found in {input_folder}!")

    return lesion_type, lesions


def add_func_pipeline(
    connectivity_analyses,
    lesion_type,
    input_spaces,
    output_spaces,
    connectome_name,
    connectome_mask,
):
    """Add Volume Functional Connectivity Pipeline to list of connectivity analyses to run.

    Parameters
    ----------
    connectivity_analyses : list
        List of connectivity analyses to run. May add a functional pipeline to the list.
    lesion_type : str
        Input ROI type. "volume" or "surface".
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectome_name : str
        Name of directory/connectome inside connectome_dir.
    connectome_mask : str
        Brain mask used. This is used for metadata only.

    Returns
    -------
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectivity_analyses : list
        Updated list of connectivity analysis pipelines to run.

    """
    VOL_CONN = config.connectomes["func_volume_connectomes"]

    if connectome_name:
        if connectome_name in VOL_CONN.keys():
            input_space = VOL_CONN[connectome_name]["input"][lesion_type]
            output_space = VOL_CONN[connectome_name]["output"]
            if input_space not in input_spaces:
                input_spaces.append(input_space)
            if output_space not in output_spaces:
                output_spaces.append(output_space)
            connectome = VOL_CONN[connectome_name]["path"]
            if (
                lesion_type == "surface"
                and connectome_name not in config.v2s_compatible_conn
            ):
                raise ValueError(
                    f"Brain Connectome {connectome_name} is incompatible with input surface-space ROIs. Please select a compatible connectome: {list(config.v2s_compatible_conn.keys())}"
                )
            elif (
                lesion_type == "surface"
                and connectome_name in config.v2s_compatible_conn
            ):
                roi_connectome_name = config.v2s_compatible_conn[connectome_name]
                roi_connectome = config.connectomes["func_surface_connectomes"][
                    roi_connectome_name
                ]["path"]
            elif lesion_type == "volume":
                roi_connectome_name = connectome_name
                roi_connectome = connectome
            cprint(
                f"I am going to use the normative volume-space connectome files stored in: {connectome}",
                "green",
                attrs=["bold"],
            )
            connectivity_analyses.append(
                {
                    "type": "Functional",
                    "sourceCoordinateSystem": input_space,
                    "coordinateSystem": output_space,
                    "space": "volume",
                    "roi_connectome_name": roi_connectome_name,
                    "roi_connectome": roi_connectome,
                    "connectome_name": connectome_name,
                    "connectome": connectome,
                    "connectome_mask": connectome_mask,
                    "tool": "connectome_quick",
                }
            )
        else:
            raise ValueError(
                f"Invalid volume connectome name. Available volume connectomes are: {list(VOL_CONN.keys())}."
            )
    else:
        cprint(
            "I am not going to generate volume-space functional connectivity maps",
            "red",
            attrs=["bold"],
        )
    return input_spaces, output_spaces, connectivity_analyses


def add_surf_pipeline(
    connectivity_analyses,
    lesion_type,
    input_spaces,
    output_spaces,
    connectome_name,
    connectome_mask,
):
    """Add Surface Functional Connectivity Pipeline to list of connectivity analyses to run.

    Parameters
    ----------
    connectivity_analyses : list
        List of connectivity analyses to run. May add a functional pipeline to the list.
    lesion_type : str
        Input ROI type. "surface" or "volume"
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectome_name : str
        Name of directory/connectome inside connectome_dir.
    connectome_mask : str
        Brain mask used. This is used for metadata only.

    Returns
    -------
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectivity_analyses : list
        Updated list of connectivity analysis pipelines to run.

    """
    SURF_CONN = config.connectomes["func_surface_connectomes"]

    if connectome_name:
        if connectome_name in SURF_CONN.keys():
            connectome = SURF_CONN[connectome_name]["path"]
            input_space = SURF_CONN[connectome_name]["input"][lesion_type]
            output_space = SURF_CONN[connectome_name]["output"]
            if input_space not in input_spaces:
                input_spaces.append(input_space)
            if output_space not in output_spaces:
                output_spaces.append(output_space)
            if (
                lesion_type == "volume"
                and connectome_name in config.s2v_compatible_conn
            ):
                roi_connectome_name = config.s2v_compatible_conn[connectome_name]
                roi_connectome = config.connectomes["func_volume_connectomes"][
                    roi_connectome_name
                ]["path"]
            elif (
                lesion_type == "volume"
                and connectome_name not in config.s2v_compatible_conn
            ):
                raise ValueError(
                    f"Brain Connectome {connectome_name} is incompatible with input volume-space ROIs. Please select a compatible connectome: {list(config.s2v_compatible_conn.keys())}"
                )
            elif lesion_type == "surface":
                roi_connectome_name = connectome_name
                roi_connectome = connectome
            cprint(
                f"I am going to use the normative surface-space connectome files stored in: {connectome}",
                "green",
                attrs=["bold"],
            )
            connectivity_analyses.append(
                {
                    "type": "Functional",
                    "sourceCoordinateSystem": input_space,
                    "coordinateSystem": output_space,
                    "space": "surface",
                    "roi_connectome_name": roi_connectome_name,
                    "roi_connectome": roi_connectome,
                    "connectome_name": connectome_name,
                    "connectome": connectome,
                    "connectome_mask": connectome_mask,
                    "tool": "connectome_quick",
                }
            )
        else:
            raise ValueError(
                f"Invalid surface connectome name. Available surface connectomes are: {list(SURF_CONN.keys())}."
            )
    else:
        cprint(
            "I am not going to generate surface-space functional connectivity maps",
            "red",
            attrs=["bold"],
        )
    return input_spaces, output_spaces, connectivity_analyses


def add_bcb_pipeline(
    connectivity_analyses,
    lesion_type,
    input_spaces,
    output_spaces,
    connectome_name,
    connectome_mask,
):
    """Add Structural Connectivity BCB Pipeline to list of connectivity analyses to run.

    Parameters
    ----------
    connectivity_analyses : list
        List of connectivity analyses to run. May add a structural pipeline to the list.
    lesion_type : str
        Input ROI type. "surface" or "volume"
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectome_name : str
        Name of directory/connectome.
    connectome_mask : str
        Brain mask used. This is used for metadata only.

    Returns
    -------
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectivity_analyses : list
        Updated list of connectivity analysis pipelines to run.

    """
    config.verify_software(["bcb_path"])
    BCB_CONN = config.connectomes["struct_bcb_connectomes"]

    if connectome_name:
        if connectome_name in BCB_CONN.keys():
            connectome = BCB_CONN[connectome_name]["path"]
            input_space = BCB_CONN[connectome_name]["input"][lesion_type]
            output_space = BCB_CONN[connectome_name]["output"]
            if input_space not in input_spaces:
                input_spaces.append(input_space)
            if output_space not in output_spaces:
                output_spaces.append(output_space)
            connectivity_analyses.append(
                {
                    "type": "Structural",
                    "coordinateSystem": output_space,
                    "sourceCoordinateSystem": input_space,
                    "space": "volume",
                    "connectome_name": connectome_name,
                    "connectome": connectome,
                    "connectome_mask": connectome_mask,
                    "tool": "BCB Disconnectome",
                }
            )
        else:
            raise ValueError(
                f"Invalid BCB connectome name. Available BCB connectomes are: {list(BCB_CONN.keys())}."
            )
        cprint(
            f"I am going to use the structural connectome files stored in: {connectome}",
            "green",
            attrs=["bold"],
        )
    else:
        cprint(
            "I am not going to generate structural BCB connectivity maps",
            "red",
            attrs=["bold"],
        )
    return input_spaces, output_spaces, connectivity_analyses


def add_lqt_pipeline(
    connectivity_analyses,
    lesion_type,
    input_spaces,
    output_spaces,
    connectome_name,
    connectome_mask,
):
    """Add Structural Connectivity LQT Pipeline to list of connectivity analyses to run.

    Parameters
    ----------
    connectivity_analyses : list
        List of connectivity analyses to run. May add a structural pipeline to the list.
    lesion_type : str
        Input ROI type. "surface" or "volume"
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectome_name : str
        Name of directory/connectome.
    connectome_mask : str
        Brain mask used. This is used for metadata only.

    Returns
    -------
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    connectivity_analyses : list
        Updated list of connectivity analysis pipelines to run.

    """
    config.verify_software(["lqt_path"])
    LQT_CONN = config.connectomes["struct_lqt_connectomes"]

    if connectome_name:
        if connectome_name in LQT_CONN.keys():
            input_space = LQT_CONN[connectome_name]["input"][lesion_type]
            output_space = LQT_CONN[connectome_name]["output"]
            if input_space not in input_spaces:
                input_spaces.append(input_space)
            if output_space not in output_spaces:
                output_spaces.append(output_space)
            connectivity_analyses.append(
                {
                    "type": "Structural",
                    "coordinateSystem": output_space,
                    "sourceCoordinateSystem": input_space,
                    "space": "volume",
                    "connectome_name": connectome_name,
                    "connectome": LQT_CONN[connectome_name]["path"],
                    "connectome_mask": connectome_mask,
                    "tool": "Lesion Quantification Toolkit",
                }
            )
            cprint(
                f"I am going to use the Lesion Quantification Toolkit with the {connectome_name} connectome",
                "green",
                attrs=["bold"],
            )
        else:
            raise ValueError(
                f"Invalid LQT connectome name. Available LQT connectomes are: {list(LQT_CONN.keys())}."
            )
    else:
        cprint(
            "I am not going to use the Lesion Quantification Toolkit",
            "red",
            attrs=["bold"],
        )
    return input_spaces, output_spaces, connectivity_analyses


def confirm_connectivity_analyses(
    connectivity_analyses, lesion_type, input_spaces, output_spaces, override=False
):
    """Reformat and print out selected connectivity pipelines for confirmation.

    Parameters
    ---------
    connectivity_analyses : list
        List of dicts containing selected pipelines. Could contain repeats.
    lesion_type : str
        Type of input ROIs. "volume" or "surface.
    input_spaces : list
        List of input spaces.
    output_spaces : list
        List of output spaces.
    override : bool
        If True, override functional connectome matching check. Allows for mismatch of
        volume and surface connectomes. (e.g. using volume yeo1000_dil connectome and
        surface GSP1000_MF_surf_fs5 connectome)

    Returns
    -------
    vol_spaces : dict of input/output lists
        Dict of input and output volume space lists.
    surf_spaces : dict of input/output lists
        Dict of input and output surface space lists.
    set_connectivity_analyses : list
        List of dicts containing selected pipelines. Potential pipeline repeats omitted.

    """
    set_connectivity_analyses = []
    fc_conn = {"volume": "", "surface": ""}
    if len(connectivity_analyses) == 0:
        raise ValueError("You have no analyses selected!")
    for i in connectivity_analyses:
        if i not in set_connectivity_analyses:
            set_connectivity_analyses.append(i)
    for a in set_connectivity_analyses:
        if a["tool"] == "connectome_quick":
            print(
                f"I will compute {a['type']} connectivity between {lesion_type} ROIs in the {a['roi_connectome_name']} connectome to {a['coordinateSystem']} {a['space']} space using the {a['connectome_name']} connectome with {a['tool']}."
            )
            fc_conn[a["space"]] = a["connectome_name"]
        else:
            print(
                f"I will compute {a['type']} connectivity in {a['coordinateSystem']} {a['space']} space using the {a['connectome_name']} connectome with {a['tool']}."
            )
    # Raise Warning if running volume-space connectome_quick and surface-space connectome_quick with non-comparable output connectomes. Does not prevent investigator from proceeding, however.
    if fc_conn["volume"] and fc_conn["surface"]:
        if (config.v2s_compatible_conn[fc_conn["volume"]] == fc_conn["surface"]) and (
            config.s2v_compatible_conn[fc_conn["surface"]] == fc_conn["volume"]
        ):
            cprint(
                "Verified: Output functional connectivity connectomes match",
                "green",
                attrs=["bold"],
            )
        else:
            cprint(
                "WARNING: Output functional connectivity connectomes do not match",
                "red",
                attrs=["bold"],
            )
            if not override:
                raise ValueError(
                    f"Output functional connectivity connectomes do not match and will probably not be comparable. You have currently selected to obtain volume-space functional connectivity outputs using the {fc_conn['volume']} connectome and surface-space functional connectivity outputs using the {fc_conn['surface']} connectome. If you wish to proceed with the selected output connectomes, set 'override' to True"
                )
    # Generate volume and surface space lists
    vol_spaces = {"input": [], "output": []}
    surf_spaces = {"input": [], "output": []}

    for space in config.volume_spaces:
        if space in input_spaces:
            vol_spaces["input"].append(space)
        if space in output_spaces:
            vol_spaces["output"].append(space)
    for space in config.surface_spaces:
        if space in input_spaces:
            surf_spaces["input"].append(space)
        if space in output_spaces:
            surf_spaces["output"].append(space)

    return vol_spaces, surf_spaces, set_connectivity_analyses


def init_meta_df(lesion_type, lesions, project_path):
    """Initialize meta_df and copy/rename input ROIs to BIDS format

    Parameters
    ----------
    lesion_type : str
        "volume" or "surface
    lesions : dict
        Dictionary containing subject names and ROI paths
    project_path : str
        Path to project directory

    Returns
    -------
    meta_df : pandas.DataFrame
        Metadata Dataframe

    """
    ipython = get_ipython()
    if lesion_type == "volume":
        volumes = []
        sub_names = []
        for l in lesions:
            sub_names.append(l)
            volumes.append(lesions[l])
        meta_df = pd.DataFrame({"source_subject": sub_names, "source_vol": volumes})
    elif lesion_type == "surface":
        lh = []
        rh = []
        sub_names = []
        for l in lesions:
            sub_names.append(l)
            lh.append(lesions[l]["lh"])
            rh.append(lesions[l]["rh"])
        meta_df = pd.DataFrame(
            {"source_subject": sub_names, "source_surf_lh": lh, "source_surf_rh": rh}
        )

    layout = BIDSLayout(
        project_path, validate=False, config=nimds.get_filepath("connectivity_config")
    )

    orig_dir = os.path.join(project_path, "inputs")
    if not os.path.exists(orig_dir):
        os.makedirs(orig_dir)

    transferred_lesion_files = []
    subject_names = []
    # Copy over lesion masks to project folder, and if Nifti, make sure they're compressed
    # by checking for '.gz' at end of filename. Gifti files remain uncompressed.
    if lesion_type == "volume":
        for i in range(len(meta_df)):
            subject_names.append(meta_df.iloc[i]["source_subject"])
            basename = os.path.basename(meta_df.iloc[i]["source_vol"])
            if basename.split(".")[-1] == "gz":
                shutil.copy(meta_df.iloc[i]["source_vol"], orig_dir)
                transferred_lesion_files.append(os.path.join(orig_dir, basename))
            else:
                with open(meta_df.iloc[i]["source_vol"], "rb") as f_in:
                    with gzip.open(
                        os.path.join(orig_dir, basename + ".gz"), "wb"
                    ) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        transferred_lesion_files.append(
                            os.path.join(orig_dir, basename + ".gz")
                        )
    elif lesion_type == "surface":
        for i in range(len(meta_df)):
            subject_names.append(meta_df.iloc[i]["source_subject"])
            basename_lh = os.path.basename(meta_df.iloc[i]["source_surf_lh"])
            basename_rh = os.path.basename(meta_df.iloc[i]["source_surf_rh"])
            shutil.copy(meta_df.iloc[i]["source_surf_lh"], orig_dir)
            shutil.copy(meta_df.iloc[i]["source_surf_rh"], orig_dir)
            transferred_lesion_files.append(
                [
                    os.path.join(orig_dir, basename_lh),
                    os.path.join(orig_dir, basename_rh),
                ]
            )

    ipython.system(f"datalad save -d {project_path} -m 'Added original files'")

    # Reformat filenames
    subjects_list = []
    bids_lesions = []

    if lesion_type == "volume":
        for subject_name, i in zip(subject_names, transferred_lesion_files):
            subject_name = nimtrack.bids_sanitize(subject_name)
            subjects_list.append(subject_name)
            new_filepath = layout.build_path(
                {
                    "subject": subject_name,
                    "datatype": "roi",
                    "coordinateSystem": "original",
                    "suffix": "lesionMask",
                },
                validate=False,
            )
            os.makedirs("/".join(new_filepath.split("/")[:-1]), exist_ok=True)
            bids_lesions.append(new_filepath)
            shutil.copy(i, new_filepath)
    elif lesion_type == "surface":
        for subject_name, i in zip(subject_names, transferred_lesion_files):
            subject_name = nimtrack.bids_sanitize(subject_name)
            subjects_list.append(subject_name)
            hemispheres = []
            for j, hemi in enumerate(["L", "R"]):
                new_filepath = layout.build_path(
                    {
                        "subject": subject_name,
                        "hemisphere": hemi,
                        "datatype": "roi",
                        "suffix": "atrophyMap",
                        "coordinateSystem": "original",
                        "extension": ".gii",
                    },
                    validate=False,
                )
                os.makedirs("/".join(new_filepath.split("/")[:-1]), exist_ok=True)
                hemispheres.append(new_filepath)
                shutil.copy(i[j], new_filepath)
            bids_lesions.append(hemispheres)

    ipython.system(
        f"datalad save -d {project_path} -m 'Reformatted lesion filenames to BIDS'"
    )

    meta_df["subject"] = subjects_list
    if lesion_type == "volume":
        meta_df["orig_vol_lesion"] = bids_lesions
    elif lesion_type == "surface":
        lh = [h[0] for h in bids_lesions]
        rh = [h[1] for h in bids_lesions]
        meta_df["orig_Lsurf_lesion"] = lh
        meta_df["orig_Rsurf_lesion"] = rh

    return meta_df


def reslice_and_convert_rois(
    lesion_type,
    meta_df,
    vol_spaces,
    surf_spaces,
    project_path,
    doThreshold=True,
    binarize=True,
    threshold=0.0,
    direction="twosided",
    RF_type="RF_ANTs",
    interp="linear",
    force_resample=False,
):
    """Reslice and convert ROIs between different coordinate spaces

    Parameters
    ----------
    lesion_type : str
        "volume" or "surface".
    meta_df : pandas.DataFrame
        Metadata dataframe.
    vol_spaces : dict of input/output lists
        Dict of input and output volume space lists.
    surf_spaces : dict of input/output lists
        Dict of input and output surface space lists.
    project_path : str
        Path to Project directory.
    doThreshold : bool
        If True, threshold ROIs. If False, leave unthresholded.
    binarize : bool
        If True, binarize ROIs. If False, leave unbinarized.
    threshold : float
        Threshold for output images. If binarize is True, use threshold to binarize images.
    direction : str
        If "less", keep values less than or equal to the threshold value.
        If "greater", keep values greater than or equal to the threshold value.
        If "twosided", keep values less than -{threshold} and values greater than {threshold}. Defaults to "twosided".
        Only applies to Surface files.
    RF_type : str
        RF_type for CBIG RegistrationFusion. See cbig_Registration_Fusion_surf2vol.
    interp : str
        Interpolation method for CBIG RegistrationFusion. See cbig_Registration_Fusion_surf2vol.
    force_resample : bool
        If True, resamples all input images to FSL MNI152 1mm or 2mm mask. Defaults to False.

    Returns
    -------
    meta_df : pandas.DataFrame
        Metadata dataframe updated with sliced/converted ROIs

    """
    config.verify_software(["fsl_path", "freesurfer_path"])
    FSLDIR = config.software["fsl_path"]
    os.environ["FREESURFER_HOME"] = config.software["freesurfer_path"]
    os.environ["SUBJECTS_DIR"] = os.getenv("FREESURFER_HOME") + "/subjects"
    os.environ["FSFAST_HOME"] = os.getenv("FREESURFER_HOME") + "/fsfast"
    os.environ["MNI_DIR"] = os.getenv("FREESURFER_HOME") + "/mni"
    tmpdir = os.path.join(project_path, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    os.environ["PATH"] = (
        FSLDIR + "/bin:" + os.getenv("FREESURFER_HOME") + "/bin:" + os.getenv("PATH")
    )
    os.environ["FSLDIR"] = FSLDIR
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    ipython = get_ipython()
    ipython.system("source $FREESURFER_HOME/SetUpFreeSurfer.sh")
    ipython.system(f". {FSLDIR}/etc/fslconf/fsl.sh")

    # Make a copy of the original lesions in orig_dir, then make conformed lesions in lesion_dir:
    layout = BIDSLayout(
        project_path, validate=False, config=nimds.get_filepath("connectivity_config")
    )

    sliced_volumes = {}
    for space in vol_spaces["input"]:
        sliced_volumes[space] = []

    sliced_surfaces = {}
    for space in surf_spaces["input"]:
        sliced_surfaces[space] = {"lh": [], "rh": []}

    for i in meta_df.itertuples():
        # For Volume ROIs:
        if lesion_type == "volume":
            ipython.system(f"datalad unlock -d {project_path} {i.orig_vol_lesion}")
            # Reslicing to 2mm/1mm volume space if needed
            for space in vol_spaces["input"]:
                # Make a copy of the lesion in the sliced directory
                new_filepath = layout.build_path(
                    {
                        "subject": i.subject,
                        "datatype": "roi",
                        "suffix": "lesionMask",
                        "coordinateSystem": space,
                    },
                    validate=False,
                )
                os.makedirs("/".join(new_filepath.split("/")[:-1]), exist_ok=True)
                sliced_volumes[space].append(new_filepath)
                shutil.copy(i.orig_vol_lesion, new_filepath)

                # Reslice the lesions according to a standard, if needed.
                if space == "2mm":
                    standard = "MNI152_T1_2mm.nii.gz"
                elif space == "1mm":
                    standard = "MNI152_T1_1mm.nii.gz"
                # denan all images, replace nan with 0
                ipython.system(
                    f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -nan {new_filepath}"
                )
                # Check original resolution of image
                img = image.load_img(new_filepath)
                # If affine matches standard MNI152 affines AND image is in the desired resolution, do not resample
                if (
                    np.all(img.affine == FSL_MNI152_2MM_AFFINE) and (space == "2mm")
                ) or (np.all(img.affine == FSL_MNI152_1MM_AFFINE) and (space == "1mm")):
                    pass
                # Else, if affine matches standard MNI152 affines, resample to desired resolution.
                # OR
                # If force_resample is True, then resample to desired resolution.
                elif (
                    (np.all(img.affine == FSL_MNI152_2MM_AFFINE) and (space == "1mm"))
                    or (
                        np.all(img.affine == FSL_MNI152_1MM_AFFINE) and (space == "2mm")
                    )
                    or force_resample
                ):
                    if force_resample:
                        cprint(
                            f"Force Resampling to FSL MNI152 {space} Template: {i.orig_vol_lesion}",
                            "red",
                        )
                    else:
                        cprint(
                            f"Resampling to FSL MNI152 {space} Template: {i.orig_vol_lesion}",
                            "green",
                        )
                    # Else, if image is categorial or binary (values are all integers), then use nearest neighbour interpolation
                    if np.sum(img.get_fdata()) == np.sum(img.get_fdata().astype(int)):
                        interp_method = "nearest"
                    # Else, if image is continuous (values are all floats), then use nearest interpolation
                    else:
                        interp_method = "continuous"
                    image.resample_to_img(
                        img, f"{FSLDIR}/data/standard/{standard}", interp_method
                    ).to_filename(new_filepath)
                # If affine does not match standard MNI152 affines, then raise Error.
                else:
                    raise ValueError(
                        "Input images must be in MNI152 space and have the same affine "
                        "as the FSL MNI152 templates! Your image has affine\n"
                        f"{img.affine}\n"
                        "but we expected an affine of\n"
                        f"{FSL_MNI152_1MM_AFFINE}\n"
                        "or\n"
                        f"{FSL_MNI152_2MM_AFFINE}\n"
                        "If you are ABSOLUTELY sure your images are registered in "
                        "MNI152 space, add the option 'force_resample=True' to "
                        "reslice_and_convert_rois"
                    )
                if doThreshold:
                    if binarize:
                        if direction == "twosided":
                            thr_img = os.path.join(tmpdir, "thr.nii.gz")
                            uthr_img = os.path.join(tmpdir, "uthr.nii.gz")
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -thr {threshold} -bin {thr_img} -odt float"
                            )
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -uthr -{threshold} -mul -1 -bin {uthr_img} -odt float"
                            )
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {thr_img} -sub {uthr_img} {new_filepath} -odt float"
                            )
                        elif direction == "greater":
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -thr {threshold} -bin {new_filepath} -odt float"
                            )
                        elif direction == "less":
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -uthr {threshold} -mul -1 -bin -mul -1 {new_filepath} -odt float"
                            )
                    else:
                        if direction == "twosided":
                            thr_img = os.path.join(tmpdir, "thr.nii.gz")
                            uthr_img = os.path.join(tmpdir, "uthr.nii.gz")
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -thr {threshold} {thr_img} -odt float"
                            )
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -uthr -{threshold} -mul -1 {uthr_img} -odt float"
                            )
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {thr_img} -sub {uthr_img} {new_filepath} -odt float"
                            )
                        elif direction == "greater":
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -thr {threshold} {thr_img} -odt float"
                            )
                        elif direction == "less":
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -uthr {threshold} {thr_img} -odt float"
                            )

        elif lesion_type == "surface":
            ipython.system(f"datalad unlock -d {project_path} {i.orig_Lsurf_lesion}")
            ipython.system(f"datalad unlock -d {project_path} {i.orig_Rsurf_lesion}")
            # Resampling to fs5 if needed
            for space in surf_spaces["input"]:
                order = space[-1]

                # Get filepaths of the converted lesion in the sliced directory
                new_filepath_lh = layout.build_path(
                    {
                        "subject": i.subject,
                        "hemisphere": "L",
                        "datatype": "roi",
                        "suffix": "atrophyMap",
                        "coordinateSystem": space,
                        "extension": ".gii",
                    },
                    validate=False,
                )
                new_filepath_rh = layout.build_path(
                    {
                        "subject": i.subject,
                        "hemisphere": "R",
                        "datatype": "roi",
                        "suffix": "atrophyMap",
                        "coordinateSystem": space,
                        "extension": ".gii",
                    },
                    validate=False,
                )
                os.makedirs("/".join(new_filepath_lh.split("/")[:-1]), exist_ok=True)
                os.makedirs("/".join(new_filepath_rh.split("/")[:-1]), exist_ok=True)
                sliced_surfaces[space]["lh"].append(new_filepath_lh)
                sliced_surfaces[space]["rh"].append(new_filepath_rh)
                shutil.copy(i.orig_Lsurf_lesion, new_filepath_lh)
                shutil.copy(i.orig_Rsurf_lesion, new_filepath_rh)

                # Denan surface files
                nib.save(nimsf.nan_to_num(nib.load(new_filepath_lh)), new_filepath_lh)
                nib.save(nimsf.nan_to_num(nib.load(new_filepath_rh)), new_filepath_rh)

                # Check Resolution
                gifti_lh = nib.load(new_filepath_lh)
                gifti_rh = nib.load(new_filepath_rh)

                if (
                    gifti_lh.agg_data().shape in FSAVERAGE_SHAPES
                    and gifti_rh.agg_data().shape in FSAVERAGE_SHAPES
                    and gifti_lh.agg_data().shape == gifti_rh.agg_data().shape
                ):
                    pass
                elif force_resample:
                    print(
                        f"Resampling to {space}: {new_filepath_lh} and {new_filepath_rh}"
                    )
                    # Resample
                    ipython.system(
                        f"{config.software['freesurfer_path']}/bin/mri_surf2surf --hemi lh --srcsubject ico --srcsurfval {i.orig_Lsurf_lesion} --trgsubject ico --trgicoorder {order} --trgsurfval {new_filepath_lh}"
                    )
                    ipython.system(
                        f"{config.software['freesurfer_path']}/bin/mri_surf2surf --hemi rh --srcsubject ico --srcsurfval {i.orig_Rsurf_lesion} --trgsubject ico --trgicoorder {order} --trgsurfval {new_filepath_rh}"
                    )
                else:
                    raise ValueError(
                        "Input images must be in fsaverage5 space! We recommend, if "
                        "possible, that you convert directly from native space to "
                        "fsaverage5. If you wish to force resample from other fsaverage"
                        " resolutions to fsaverage5 space, add the option "
                        "'force_resample=True' to reslice_and_convert_rois"
                    )

                # Threshold (and binarize)
                if doThreshold:
                    lh_gifti = nib.load(new_filepath_lh)
                    rh_gifti = nib.load(new_filepath_rh)
                    lh_gifti = nimsf.threshold(
                        lh_gifti, threshold, direction, binarize, 0
                    )
                    nib.save(lh_gifti, new_filepath_lh)
                    rh_gifti = nimsf.threshold(
                        rh_gifti, threshold, direction, binarize, 0
                    )
                    nib.save(rh_gifti, new_filepath_rh)

            # Convert surface ROIs to volume space with CBIG RegistrationFusion
            if len(vol_spaces["input"]) > 0:
                # Generate temporary volume nifti file
                temp_vol_nifti = cbig_RegistrationFusion_surf2vol(
                    lh=new_filepath_lh,
                    rh=new_filepath_rh,
                    subject_name="TMP_" + i.subject,
                    tmpdir=tmpdir,
                    RF_type=RF_type,
                    interp=interp,
                    threshold=threshold,
                    binarize=binarize,
                )
                # Resample temporary volume nifti file to desired volume coordinate spaces
                for space in vol_spaces["input"]:
                    print(
                        f"Resampling to FSL MNI152 {space} Template: {i.orig_Lsurf_lesion} and {i.orig_Rsurf_lesion}"
                    )
                    # Get filepaths of the converted lesion in the sliced directory
                    new_filepath = layout.build_path(
                        {
                            "subject": i.subject,
                            "datatype": "roi",
                            "suffix": "lesionMask",
                            "coordinateSystem": space,
                        },
                        validate=False,
                    )
                    os.makedirs("/".join(new_filepath.split("/")[:-1]), exist_ok=True)
                    sliced_volumes[space].append(new_filepath)
                    shutil.copy(temp_vol_nifti, new_filepath)

                    # Reslice the lesions according to a standard
                    if space == "2mm":
                        standard = "MNI152_T1_2mm.nii.gz"
                    elif space == "1mm":
                        standard = "MNI152_T1_1mm.nii.gz"
                    image.resample_to_img(
                        new_filepath, f"{FSLDIR}/data/standard/{standard}", "continuous"
                    ).to_filename(new_filepath)
                    if doThreshold:
                        if binarize:
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -thr {threshold} -bin {new_filepath} -odt float"
                            )
                        else:
                            ipython.system(
                                f"{config.software['fsl_path']}/bin/fslmaths {new_filepath} -thr {threshold} {new_filepath} -odt float"
                            )

    if lesion_type == "volume":
        for space in vol_spaces["input"]:
            meta_df[f"lesion_{space}"] = sliced_volumes[space]
    elif lesion_type == "surface":
        for space in vol_spaces["input"]:
            meta_df[f"lesion_{space}"] = sliced_volumes[space]
        for space in surf_spaces["input"]:
            meta_df[f"lesion_{space}_lh"] = sliced_surfaces[space]["lh"]
            meta_df[f"lesion_{space}_rh"] = sliced_surfaces[space]["rh"]

    ipython.system(
        f"datalad save -d {project_path} -m 'Resliced, converted, and binarized lesions'"
    )

    return meta_df


def cbig_RegistrationFusion_surf2vol(
    lh,
    rh,
    subject_name,
    tmpdir,
    RF_type="RF_ANTs",
    interp="linear",
    threshold=0.0,
    binarize=True,
):
    """Run CBIG RegistrationFusion surf2vol on the notebook host. Generates a 2mm nifti file.

    Parameters
    ----------
    lh : str
        Path to left hemisphere gifti file.
    rh : str
        Path to right hemisphere gifti file.
    subject_name : str
        Subject Identifier.
    tmpdir : str
        Path to scratch directory.
    RF_type : str
        Type of Registration Fusion approaches used to generate the mappings ("RF_M3Z" or "RF_ANTs"). RF-M3Z is recommended if data was registered from subject's space to the volumetric atlas space using FreeSurfer. RF-ANTs is recommended if such registrations were carried out using other tools, especially ANTs. Defaults to "RF_ANTs".
    interp : str
        Interpolation method ("linear" or "nearest"). Defaults to "linear".
    threshold : float
        Threshold for output image.
    binarize : bool
        If True, binarize output image with threshold.

    Returns
    -------
    outfile : str
        Path to output converted nifti file.

    Outputs
    -------
    Generates a 2mm volume nifti file with filename {tmpdir}/tmp_cbigrf/{subject_name}.nii.gz

    """
    config.verify_software(["cbig_path", "freesurfer_path"])
    # Set environment variables
    os.environ["FREESURFER_HOME"] = config.software["freesurfer_path"]
    os.environ["SUBJECTS_DIR"] = os.getenv("FREESURFER_HOME") + "/subjects"
    os.environ["FSFAST_HOME"] = os.getenv("FREESURFER_HOME") + "/fsfast"
    os.environ["MNI_DIR"] = os.getenv("FREESURFER_HOME") + "/mni"
    os.environ["TMPDIR"] = tmpdir
    os.environ["PATH"] = os.getenv("FREESURFER_HOME") + "/bin:" + os.getenv("PATH")
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    os.environ["NIMLAB_CBIG_PATH"] = config.software["cbig_path"]
    os.makedirs(os.path.join(tmpdir, "tmp_cbigrf"), exist_ok=True)

    # Generate matfiles and csv
    lh_img, rh_img = nib.load(lh).agg_data(), nib.load(rh).agg_data()
    sub = {"lh_label": lh_img, "rh_label": rh_img}
    matfile_path = os.path.join(tmpdir, "tmp_cbigrf", subject_name + ".mat")
    savemat(matfile_path, sub)
    mat_files = [matfile_path]

    mat_file_list_path = os.path.join(tmpdir, "tmp_cbigrf", "matfile_list.csv")
    mat_files = pd.DataFrame(mat_files)
    mat_files.to_csv(mat_file_list_path, index=False, header=False)

    # Run CBIG RegistrationFusion
    script_path = os.path.join(
        config.software["cbig_path"],
        "stable_projects/registration/Wu2017_RegistrationFusion/bin/scripts_final_proj/CBIG_RF_projectfsaverage2Vol_batch.sh",
    )
    descriptor = "allSub_fsaverage_to_FSL_MNI152_FS4.5.0_" + RF_type
    connectome_cmd = [
        "sh",
        script_path,
        "-l",
        mat_file_list_path,
        "-o",
        os.path.join(tmpdir, "tmp_cbigrf"),
        "-r",
        RF_type,
        "-i",
        interp,
    ]
    ipython = get_ipython()
    ipython.system("source $FREESURFER_HOME/SetUpFreeSurfer.sh")
    ipython.system(" ".join(connectome_cmd))

    vol_img_path = os.path.join(
        tmpdir, "tmp_cbigrf", subject_name + "." + descriptor + ".nii.gz"
    )
    outfile = os.path.join(tmpdir, "tmp_cbigrf", subject_name + ".nii.gz")

    if binarize:
        vol_img = image.new_img_like(
            vol_img_path, (image.get_data(vol_img_path) > threshold)
        )
    else:
        thresholded_image = image.threshold_img(image.load_img(vol_img_path), threshold)
        vol_img = image.new_img_like(vol_img_path, thresholded_image.get_fdata())

    image.resample_to_img(
        vol_img,
        nimds.get_img("MNI152_T1_2mm_brain_mask_dil"),
        interpolation="continuous",
    ).to_filename(outfile)

    return outfile


def review_lesions(
    visualize,
    lesion_type,
    meta_df,
    project_path,
    vol_spaces,
    surf_spaces,
    brain_mask_2mm,
    brain_mask_1mm,
):

    ipython = get_ipython()
    # Load volume lesion images and remove excess frames (4th dimension) if needed
    for space in vol_spaces["input"]:
        for l in tqdm(meta_df[f"lesion_{space}"]):
            lesion_img = image.load_img(l)
            if len(lesion_img.shape) > 3:
                lesion_img = image.index_img(lesion_img, 0)
                ipython.system(f"datalad unlock -d {project_path} {l}")
                lesion_img.to_filename(l)
    ipython.system(
        f"datalad save -d {project_path} -m 'Removed excess 4th dimension frames'"
    )

    nonbrain_mask_2mm = image.math_img("-img+1", img=brain_mask_2mm)
    nonbrain_mask_1mm = image.math_img("-img+1", img=brain_mask_1mm)

    brain_masks = {
        "1mm": {"nonbrain": nonbrain_mask_1mm, "brain": brain_mask_1mm},
        "2mm": {"nonbrain": nonbrain_mask_2mm, "brain": brain_mask_2mm},
    }
    need_check = {}
    for space in vol_spaces["input"]:
        need_check[space] = []

    # Analyze the lesions
    for i in meta_df.iterrows():
        for space in vol_spaces["input"]:
            lesion_in_brain = image.math_img(
                "img1 * img2",
                img1=i[1][f"lesion_{space}"],
                img2=brain_masks[space]["brain"],
            )
            lesion_outside_of_brain = image.math_img(
                "img1 * img2",
                img1=i[1][f"lesion_{space}"],
                img2=brain_masks[space]["nonbrain"],
            )
            voxels_in_brain = np.count_nonzero(
                np.nan_to_num(lesion_in_brain.get_fdata())
            )
            voxels_outside_of_brain = np.count_nonzero(
                np.nan_to_num(lesion_outside_of_brain.get_fdata())
            )
            combined_lesion = image.math_img(
                "img1 - img2", img1=lesion_in_brain, img2=lesion_outside_of_brain
            )
            cprint(
                i[1]["subject"]
                + " has "
                + str(voxels_in_brain)
                + " voxels inside and "
                + str(voxels_outside_of_brain)
                + " voxels outside of the "
                + str(space)
                + " brain mask",
                "green",
            )
            subject_name = i[1]["subject"]

            if voxels_in_brain == 0:
                cprint(
                    "WARNING: Map is completely blank inside the brain mask",
                    "red",
                    attrs=["bold"],
                )
                need_check[space].append(True)

            elif voxels_outside_of_brain > 0.02 * (
                voxels_in_brain + voxels_outside_of_brain
            ):
                cprint(
                    "WARNING: Greater than 2% of voxels of lesion are outside of brain mask",
                    "red",
                    attrs=["bold"],
                )
                need_check[space].append(True)
            else:
                need_check[space].append(False)

            if visualize:
                plotting.plot_glass_brain(
                    combined_lesion,
                    display_mode="lyrz",
                    title=subject_name,
                    cmap="bwr",
                    plot_abs=False,
                    colorbar=False,
                )
                plotting.plot_stat_map(
                    combined_lesion,
                    display_mode="z",
                    cut_coords=12,
                    cmap="bwr",
                    colorbar=False,
                )
                plt.show()

    for space in vol_spaces["input"]:
        meta_df[f"need_check_{space}"] = need_check[space]
        if lesion_type == "volume":
            display(
                meta_df[
                    [
                        "subject",
                        "orig_vol_lesion",
                        f"lesion_{space}",
                        f"need_check_{space}",
                    ]
                ][meta_df[f"need_check_{space}"] == True]
            )
        elif lesion_type == "surface":
            display(
                meta_df[["subject", f"lesion_{space}", f"need_check_{space}"]][
                    meta_df[f"need_check_{space}"] == True
                ]
            )
    return meta_df, brain_masks


def trim_lesions(meta_df, vol_spaces, surf_spaces, project_path, brain_masks):
    """Trim Volume space Lesions to brain mask"""
    ipython = get_ipython()
    # Trim lesions with brain_mask:
    for i in meta_df.iterrows():
        for space in vol_spaces["input"]:
            lesion_path = i[1][f"lesion_{space}"]
            ipython.system(f"datalad unlock -d {project_path} {lesion_path}")
            lesion_in_brain = image.math_img(
                "img1 * img2", img1=lesion_path, img2=brain_masks[space]["brain"]
            )
            lesion_in_brain.to_filename(lesion_path)
            print("Printed trimmed version of", os.path.basename(lesion_path))

    ipython.system(f"datalad save -d {project_path} -m 'Trimmmed lesions'")

    return meta_df


def show_lesion_overview(meta_df, vol_spaces, surf_spaces):
    """Show Volume Lesion Overview

    TODO: plot a Surface ROI overview

    """
    # Generate an "N-image" and a "Coverage Map", then show me:
    for space in vol_spaces["input"]:
        lesions = [nib.load(img) for img in meta_df[f"lesion_{space}"]]
        lesion_overlap = sum_imgs(lesions)
        lesion_mask = image.math_img("img>=1", img=lesion_overlap)

        plotting.plot_stat_map(
            lesion_overlap,
            display_mode="z",
            title=f"{space} maps",
            cut_coords=range(-54, 72, 6),
            cmap="Spectral_r",
            colorbar=False,
        )
        plotting.plot_stat_map(
            lesion_mask,
            display_mode="z",
            cut_coords=range(-54, 72, 6),
            cmap="bwr",
            colorbar=False,
        )


def generate_roi_json_sidecars(
    meta_df, vol_spaces, surf_spaces, project_path, lesion_type
):
    """Generate JSON sidecars for ROI files"""
    ipython = get_ipython()
    layout = BIDSLayout(
        project_path, validate=False, config=nimds.get_filepath("connectivity_config")
    )
    for i in tqdm(meta_df.iterrows()):
        if lesion_type == "volume":
            lesion_hash = hash_file(i[1]["orig_vol_lesion"])
            json_path = layout.build_path(
                {
                    "subject": i[1]["subject"],
                    "datatype": "roi",
                    "suffix": "lesionMask",
                    "extension": "json",
                    "coordinateSystem": "original",
                },
                validate=False,
            )
            with open(json_path, "w+") as json_out:
                json_out.write(json.dumps({"md5": lesion_hash}, indent=1))
        elif lesion_type == "surface":
            lh_lesion_hash = hash_file(i[1]["orig_Lsurf_lesion"])
            rh_lesion_hash = hash_file(i[1]["orig_Rsurf_lesion"])
            lh_json_path = layout.build_path(
                {
                    "subject": i[1]["subject"],
                    "hemisphere": "L",
                    "datatype": "roi",
                    "suffix": "atrophyMap",
                    "extension": "json",
                    "coordinateSystem": "original",
                },
                validate=False,
            )
            rh_json_path = layout.build_path(
                {
                    "subject": i[1]["subject"],
                    "hemisphere": "R",
                    "datatype": "roi",
                    "suffix": "atrophyMap",
                    "extension": "json",
                    "coordinateSystem": "original",
                },
                validate=False,
            )
            with open(lh_json_path, "w+") as json_out:
                json_out.write(json.dumps({"md5": lh_lesion_hash}, indent=1))
            with open(rh_json_path, "w+") as json_out:
                json_out.write(json.dumps({"md5": rh_lesion_hash}, indent=1))

        for space in vol_spaces["input"]:
            lesion_size = np.count_nonzero(
                nib.load(i[1]["lesion_" + space]).get_fdata()
            )
            lesion_hash = hash_file(i[1][f"lesion_{space}"])
            json_path = layout.build_path(
                {
                    "subject": i[1]["subject"],
                    "datatype": "roi",
                    "suffix": "lesionMask",
                    "extension": "json",
                    "coordinateSystem": space,
                },
                validate=False,
            )
            with open(json_path, "w+") as json_out:
                json_out.write(
                    json.dumps({"roi_size": lesion_size, "md5": lesion_hash}, indent=1)
                )
        for space in surf_spaces["input"]:
            if lesion_type == "surface":
                lh_lesion_size = np.count_nonzero(
                    nib.load(i[1][f"lesion_{space}_lh"]).agg_data()
                )
                rh_lesion_size = np.count_nonzero(
                    nib.load(i[1][f"lesion_{space}_rh"]).agg_data()
                )
                lh_lesion_hash = hash_file(i[1][f"lesion_{space}_lh"])
                rh_lesion_hash = hash_file(i[1][f"lesion_{space}_rh"])
                lh_json_path = layout.build_path(
                    {
                        "subject": i[1]["subject"],
                        "hemisphere": "L",
                        "datatype": "roi",
                        "suffix": "atrophyMap",
                        "extension": "json",
                        "coordinateSystem": space,
                    },
                    validate=False,
                )
                rh_json_path = layout.build_path(
                    {
                        "subject": i[1]["subject"],
                        "hemisphere": "R",
                        "datatype": "roi",
                        "suffix": "atrophyMap",
                        "extension": "json",
                        "coordinateSystem": space,
                    },
                    validate=False,
                )
                with open(lh_json_path, "w+") as json_out:
                    json_out.write(
                        json.dumps(
                            {"roi_size": lh_lesion_size, "md5": lh_lesion_hash},
                            indent=1,
                        )
                    )
                with open(rh_json_path, "w+") as json_out:
                    json_out.write(
                        json.dumps(
                            {"roi_size": rh_lesion_size, "md5": rh_lesion_hash},
                            indent=1,
                        )
                    )

    ipython.system(f"datalad save -d {project_path} -m 'Added lesion sidecars'")


def generate_cleaned_roi_lists(set_connectivity_analyses, meta_df, project_path):
    """Generate Cleaned ROI CSV lists"""
    for analysis in set_connectivity_analyses:
        lesion_list = os.path.join(
            project_path, f"lesion_list_{analysis['sourceCoordinateSystem']}.csv"
        )
        analysis["input_list"] = lesion_list
        if analysis["sourceCoordinateSystem"] in ["2mm", "1mm"]:
            np.savetxt(
                lesion_list,
                meta_df[f'lesion_{analysis["sourceCoordinateSystem"]}'],
                delimiter=",",
                fmt="%s",
            )
        elif analysis["sourceCoordinateSystem"] in ["fs5"]:
            lh_surface_rois = meta_df[f'lesion_{analysis["sourceCoordinateSystem"]}_lh']
            rh_surface_rois = meta_df[f'lesion_{analysis["sourceCoordinateSystem"]}_rh']
            surface_rois = np.stack([lh_surface_rois, rh_surface_rois], axis=1)
            np.savetxt(lesion_list, surface_rois, delimiter=",", fmt="%s")
        os.makedirs(os.path.join(project_path, "fc_temp"), exist_ok=True)
    return set_connectivity_analyses


def organize_connectivity_output(
    meta_df, set_connectivity_analyses, lesion_type, project_path, lqt_options
):
    """Organize Functional/Structural Connectivity Output

    If conn_quick surface output, fnames will be (lh./rh.)<subject_name>_<stat>.gii.
    Depending of original lesion type, outputs will either have _lesionMask (volume input) or _atrophyMap (surface input) in the filename.

    """
    ipython = get_ipython()
    layout = BIDSLayout(
        project_path, validate=False, config=nimds.get_filepath("connectivity_config")
    )
    for i in meta_df.iterrows():
        for analysis in set_connectivity_analyses:
            if analysis["tool"] == "connectome_quick":
                if analysis["space"] == "volume":
                    for stat in [("AvgR_Fz", "avgRFz"), ("AvgR", "avgR"), ("T", "t")]:
                        if lesion_type == "volume":
                            old_filepath = os.path.join(
                                project_path,
                                "fc_temp/sub-"
                                + i[1]["subject"]
                                + "_space-"
                                + analysis["sourceCoordinateSystem"]
                                + "_lesionMask_"
                                + stat[0]
                                + ".nii.gz",
                            )
                        elif lesion_type == "surface":
                            old_filepath = os.path.join(
                                project_path,
                                "fc_temp/sub-"
                                + i[1]["subject"]
                                + "_space-"
                                + analysis["sourceCoordinateSystem"]
                                + "_atrophyMap_"
                                + stat[0]
                                + ".nii.gz",
                            )
                        new_filepath = layout.build_path(
                            {
                                "subject": i[1]["subject"],
                                "connectome": nimtrack.bids_sanitize(
                                    analysis["connectome_name"]
                                ),
                                "datatype": "connectivity",
                                "coordinateSystem": analysis["coordinateSystem"],
                                "statistic": stat[1],
                            },
                            validate=False,
                        )
                        os.makedirs(
                            "/".join(new_filepath.split("/")[:-1]), exist_ok=True
                        )
                        os.rename(old_filepath, new_filepath)
                        # Generate json sidecar
                        json_filepath = layout.build_path(
                            {
                                "subject": i[1]["subject"],
                                "connectome": nimtrack.bids_sanitize(
                                    analysis["connectome_name"]
                                ),
                                "datatype": "connectivity",
                                "statistic": stat[1],
                                "coordinateSystem": analysis["coordinateSystem"],
                                "extension": "json",
                            },
                            validate=False,
                        )
                        with open(json_filepath, "w+") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "mask": analysis["connectome_mask"],
                                        "md5": hash_file(new_filepath),
                                    },
                                    indent=1,
                                )
                            )
                elif analysis["space"] == "surface":
                    for stat in [("AvgR_Fz", "AvgRFz"), ("AvgR", "AvgR"), ("T", "T")]:
                        for hemisphere in [("L", "lh."), ("R", "rh.")]:
                            if lesion_type == "volume":
                                old_filepath = os.path.join(
                                    project_path,
                                    "fc_temp/"
                                    + hemisphere[1]
                                    + "sub-"
                                    + i[1]["subject"]
                                    + "_space-"
                                    + analysis["sourceCoordinateSystem"]
                                    + "_lesionMask_"
                                    + stat[0]
                                    + ".gii",
                                )
                            elif lesion_type == "surface":
                                old_filepath = os.path.join(
                                    project_path,
                                    "fc_temp/"
                                    + hemisphere[1]
                                    + "sub-"
                                    + i[1]["subject"]
                                    + "_space-"
                                    + analysis["sourceCoordinateSystem"]
                                    + "_atrophyMap_"
                                    + stat[0]
                                    + ".gii",
                                )
                            new_filepath = layout.build_path(
                                {
                                    "subject": i[1]["subject"],
                                    "hemisphere": hemisphere[0],
                                    "connectome": nimtrack.bids_sanitize(
                                        analysis["connectome_name"]
                                    ),
                                    "datatype": "connectivity",
                                    "coordinateSystem": analysis["coordinateSystem"],
                                    "statistic": "surf" + hemisphere[0] + "h" + stat[1],
                                    "extension": ".gii",
                                },
                                validate=False,
                            )
                            os.makedirs(
                                "/".join(new_filepath.split("/")[:-1]), exist_ok=True
                            )
                            os.rename(old_filepath, new_filepath)
                            # Generate json sidecar
                            json_filepath = layout.build_path(
                                {
                                    "subject": i[1]["subject"],
                                    "hemisphere": hemisphere[0],
                                    "connectome": nimtrack.bids_sanitize(
                                        analysis["connectome_name"]
                                    ),
                                    "datatype": "connectivity",
                                    "statistic": "surf" + hemisphere[0] + "h" + stat[1],
                                    "coordinateSystem": analysis["coordinateSystem"],
                                    "extension": "json",
                                },
                                validate=False,
                            )
                            with open(json_filepath, "w+") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "mask": analysis["connectome_mask"],
                                            "md5": hash_file(new_filepath),
                                        },
                                        indent=1,
                                    )
                                )
            elif analysis["tool"] == "BCB Disconnectome":
                for stat in [("bcb_disco", "bcbDisco")]:
                    old_filepath = os.path.join(
                        project_path,
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space-"
                        + analysis["sourceCoordinateSystem"]
                        + "_lesionMask_"
                        + stat[0]
                        + ".nii.gz",
                    )
                    new_filepath = layout.build_path(
                        {
                            "subject": i[1]["subject"],
                            "connectome": nimtrack.bids_sanitize(
                                analysis["connectome_name"]
                            ),
                            "datatype": "connectivity",
                            "coordinateSystem": analysis["coordinateSystem"],
                            "statistic": stat[1],
                        },
                        validate=False,
                    )
                    os.makedirs("/".join(new_filepath.split("/")[:-1]), exist_ok=True)
                    os.rename(old_filepath, new_filepath)
                    # Generate json sidecar
                    json_filepath = layout.build_path(
                        {
                            "subject": i[1]["subject"],
                            "connectome": nimtrack.bids_sanitize(
                                analysis["connectome_name"]
                            ),
                            "datatype": "connectivity",
                            "statistic": stat[1],
                            "coordinateSystem": analysis["coordinateSystem"],
                            "extension": "json",
                        },
                        validate=False,
                    )
                    with open(json_filepath, "w+") as f:
                        f.write(
                            json.dumps(
                                {
                                    "mask": analysis["connectome_mask"],
                                    "md5": hash_file(new_filepath),
                                },
                                indent=1,
                            )
                        )
            elif analysis["tool"] == "Lesion Quantification Toolkit":
                for stat in [
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Damage/*_percent_damage.nii.gz",
                        "parcelDmgMap",
                        ".nii.gz",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Damage/*_percent_damage.mat",
                        "parcelDmgArr",
                        ".mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Tract_Disconnection/*_percent_discon_tracts.mat",
                        "tractDisconArr",
                        ".mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*.trk.gz",
                        "parcelDisconRawTrk",
                        ".trk.gz",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*.connectivity.mat",
                        "parcelDisconRawMat",
                        ".connectivity.mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*_percent_parcel_SDC.mat",
                        "parcelDisconPercentMat",
                        ".mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*_percent_parcel_spared_SC.mat",
                        "parcelDisconPercentSparedMat",
                        ".mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*.connectogram.txt",
                        "parcelDisconConnectogram",
                        ".connectogram.txt",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*_percent_parcel_SDC.node",
                        "parcelDisconNode",
                        ".node",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*_percent_parcel_SDC.edge",
                        "parcelDisconEdge",
                        ".edge",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*_percent_parcel_spared_SC.node",
                        "parcelDisconSparedNode",
                        ".node",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*_percent_parcel_spared_SC.edge",
                        "parcelDisconSparedEdge",
                        ".edge",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*.network_measures.txt",
                        "parcelDisconNetworkMeasures",
                        ".txt",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_Disconnection/*.trk.gz.tdi.nii.gz",
                        "parcelDisconTDI",
                        ".trk.gz.tdi.nii.gz",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Disconnection_Maps/*.trk.gz",
                        "disconMapsDisconnectedStreamlineMap",
                        ".trk.gz",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Disconnection_Maps/*.trk.gz.tdi.nii.gz",
                        "disconMapsTDIFiberMap",
                        ".trk.gz.tdi.nii.gz",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Disconnection_Maps/*_percent_tdi.nii.gz",
                        "disconMapsPercentFiberMap",
                        ".nii.gz",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_SSPL/*[!delta]_SSPL_matrix.mat",
                        "parcelSSPLMat",
                        ".mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_SSPL/*_delta_SSPL_matrix.mat",
                        "parcelSSPLDeltaMat",
                        ".mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_SSPL/*_delta_SSPL.node",
                        "parcelSSPLNode",
                        ".node",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_SSPL/*_delta_SSPL.edge",
                        "parcelSSPLEdge",
                        ".edge",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_SSPL/*_indirect_SDC.mat",
                        "parcelSSPLIndirectMat",
                        ".mat",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_SSPL/*_indirect_SDC.node",
                        "parcelSSPLIndirectNode",
                        ".node",
                    ),
                    (
                        "fc_temp/sub-"
                        + i[1]["subject"]
                        + "_space*/Parcel_SSPL/*_indirect_SDC.edge",
                        "parcelSSPLIndirectEdge",
                        ".edge",
                    ),
                ]:
                    old_filepaths = glob(os.path.join(project_path, stat[0]))
                    if len(old_filepaths) == 1:
                        old_filepath = old_filepaths[0]
                        new_filepath = layout.build_path(
                            {
                                "subject": i[1]["subject"],
                                "connectome": nimtrack.bids_sanitize(
                                    analysis["connectome_name"]
                                ),
                                "datatype": "connectivity",
                                "coordinateSystem": analysis["coordinateSystem"],
                                "statistic": stat[1],
                                "extension": stat[2],
                            },
                            validate=False,
                        )
                        os.makedirs(
                            "/".join(new_filepath.split("/")[:-1]), exist_ok=True
                        )
                        os.rename(old_filepath, new_filepath)
                        # Generate json sidecar
                        json_filepath = layout.build_path(
                            {
                                "subject": i[1]["subject"],
                                "connectome": nimtrack.bids_sanitize(
                                    analysis["connectome_name"]
                                ),
                                "datatype": "connectivity",
                                "statistic": stat[1],
                                "coordinateSystem": analysis["coordinateSystem"],
                                "extension": "json",
                            },
                            validate=False,
                        )
                        with open(json_filepath, "w+") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "mask": analysis["connectome_mask"],
                                        "connectivity_type": lqt_options[
                                            "connectivity_type"
                                        ],
                                        "sspl_spared_thresh": lqt_options[
                                            "sspl_spared_thresh"
                                        ],
                                        "smooth": lqt_options["smooth"],
                                        "md5": hash_file(new_filepath),
                                    },
                                    indent=1,
                                )
                            )
                    else:
                        print(
                            f"Warning: Subject {i[1]['subject']} is ambiguously named!"
                        )
                        print(old_filepaths)

    fc_temp_dir = os.path.join(project_path, "fc_temp")
    tmp_roi_dir = os.path.join(project_path, "tmp")
    if os.path.exists(fc_temp_dir):
        shutil.rmtree(fc_temp_dir)

    ipython.system(
        f"datalad save -d {project_path} -m 'Added functional/structural connectivity maps'"
    )

    return meta_df


def generate_readme(
    meta_df,
    set_connectivity_analyses,
    project_path,
    project_name,
    creator_email,
    input_folder,
    lesions,
    lesion_type,
):
    """Generate README"""
    ipython = get_ipython()
    # Make README.md
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(os.path.join(project_path, "README.md"), "w") as f:
        sys.stdout = f  # Change the standard output to the file
        print("# README for Dataset:", project_name)
        print("")
        print("## Creation Date:", date.today())
        print("")
        print("### This Dataset was created by:", creator_email, "  ")
        print(
            "Using a notebook from the [NIMLAB notebook templates](https://github.com/nimlab/templates)"
        )
        print("")
        print(f"The {lesion_type} ROIs were imported from:", input_folder, "  ")
        print("Example path:")
        print(lesions.popitem())
        print("")
        for a in set_connectivity_analyses:
            print(
                f"{a['type']} connectivity was computed in {a['coordinateSystem']} {a['space']} space using the {a['connectome_name']} connectome with {a['tool']}."
            )
            print("")
        print("")
        sys.stdout = original_stdout  # Reset the standard output
    cprint(
        "Created " + os.path.join(project_path, "README.md"),
        "green",
        attrs=["underline", "bold"],
    )
    print("")
    print(open(os.path.join(project_path, "README.md"), "r").read())

    # Make dataset_description.json
    with open(nimds.get_filepath("dataset_description_template")) as f:
        desc = json.load(f)

    desc["Name"] = project_name
    desc["authors"].append(creator_email)
    desc["contacts"].append(creator_email)
    desc["creation_date"] = str(date.today())

    with open(os.path.join(project_path, "dataset_description.json"), "w+") as f:
        f.write(json.dumps(desc, indent=1))

    ipython.system(
        f"datalad save -d {project_path} -m 'Added readme and dataset description'"
    )


def upload_to_dl_archive(
    project_path, project_name, vol_spaces, surf_spaces, lesion_type
):
    """Upload dataset to dl_archive and update database"""
    time.sleep(10)
    ipython = get_ipython()
    ipython.system(
        f"datalad clone {project_path} {config.database['dl_archive']['path']}/{project_name}"
    )
    ipython.system(
        f"datalad get -d {config.database['dl_archive']['path']}/{project_name}"
    )
    update_mongo_dataset(
        os.path.join(f"{config.database['dl_archive']['path']}", project_name),
        config.database["dl_archive"]["mongodb_address"],
    )

    # Generate ready-to-use csv within the dataset for safety
    archive_df = BIDSLayout(
        os.path.join(f"{config.database['dl_archive']['path']}", project_name),
        validate=False,
        config=nimds.get_filepath("connectivity_config"),
    ).to_df()
    conn_files = archive_df[
        (~archive_df["statistic"].isna())
        & (
            (archive_df["extension"] == ".nii.gz")
            | (archive_df["extension"] == ".gii")
            | (archive_df["extension"] == ".mat")
            | (archive_df["extension"] == ".trk.gz")
            | (archive_df["extension"] == ".connectivity.mat")
            | (archive_df["extension"] == ".txt")
            | (archive_df["extension"] == ".connectogram.txt")
            | (archive_df["extension"] == ".node")
            | (archive_df["extension"] == ".edge")
            | (archive_df["extension"] == ".trk.gz.tdi.nii.gz")
        )
    ]
    conn_csv = pd.DataFrame()
    conn_csv["subject"] = conn_files["subject"].unique()

    # Get original ROI columns
    if lesion_type == "volume":
        conn_csv["orig_roi_vol"] = list(
            archive_df["path"][
                (archive_df["datatype"] == "roi")
                & (
                    (archive_df["extension"] == ".nii.gz")
                    | (archive_df["extension"] == ".gii")
                )
                & (archive_df["coordinateSystem"] == "original")
            ]
        )
    elif lesion_type == "surface":
        conn_csv["orig_roi_surf_lh"] = list(
            archive_df["path"][
                (archive_df["datatype"] == "roi")
                & (
                    (archive_df["extension"] == ".nii.gz")
                    | (archive_df["extension"] == ".gii")
                )
                & (archive_df["coordinateSystem"] == "original")
                & (archive_df["hemisphere"] == "L")
            ]
        )
        conn_csv["orig_roi_surf_rh"] = list(
            archive_df["path"][
                (archive_df["datatype"] == "roi")
                & (
                    (archive_df["extension"] == ".nii.gz")
                    | (archive_df["extension"] == ".gii")
                )
                & (archive_df["coordinateSystem"] == "original")
                & (archive_df["hemisphere"] == "R")
            ]
        )

    # Get resliced/resampled ROI columns
    for space in vol_spaces["input"]:
        conn_csv[f"roi_{space}"] = list(
            archive_df["path"][
                (archive_df["datatype"] == "roi")
                & (archive_df["extension"] == ".nii.gz")
                & (archive_df["coordinateSystem"] == space)
            ]
        )
    for space in surf_spaces["input"]:
        for hemisphere in ["L", "R"]:
            conn_csv[f"roi_{space}_{hemisphere}"] = list(
                archive_df["path"][
                    (archive_df["datatype"] == "roi")
                    & (archive_df["extension"] == ".gii")
                    & (archive_df["coordinateSystem"] == space)
                    & (archive_df["hemisphere"] == hemisphere)
                ]
            )

    for s in conn_files["statistic"].unique():
        files = list(
            conn_files["path"][
                (conn_files["statistic"] == s) & (conn_files["extension"] != ".json")
            ]
        )
        conn_csv[s] = files
    conn_csv.to_csv(
        os.path.join(
            f"{config.database['dl_archive']['path']}", project_name, "filelist.csv"
        ),
        index=False,
    )

    ipython.system(
        f"datalad save -d {config.database['dl_archive']['path']}/{project_name} -m 'Added file list'"
    )
