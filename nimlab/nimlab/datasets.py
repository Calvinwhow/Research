import nibabel as nib

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

datasets = {
    "MNI152_T1_1mm_brain": "MNI152_T1_1mm_brain.nii.gz",
    "MNI152_T1_1mm_brain_mask": "MNI152_T1_1mm_brain_mask.nii",
    "MNI152_T1_1mm_brain_mask_dil": "MNI152_T1_1mm_brain_mask_dil.nii.gz",
    "MNI152_T1_2mm_brain": "MNI152_T1_2mm_brain.nii",
    "MNI152_T1_2mm_brain_mask": "MNI152_T1_2mm_brain_mask.nii",
    "MNI152_T1_2mm_brain_mask_dil": "MNI152_T1_2mm_brain_mask_dil.nii.gz",
    "MNI152_T1_2mm_brain_mask_dil1": "MNI152_T1_2mm_brain_mask_dil1.nii.gz",
    "mni_icbm152": "mni_icbm152_csf_tal_nlin_asym_09a_bin_08_inv_rl_bin_075_brain_masked.nii.gz",
    "222": "222.nii.gz",
    "222_diff": "222_diff.nii.gz",
    "dil_diff": "dil_diff.nii.gz",
    "dil1_diff": "dil1_diff.nii.gz",
    "fs5_mask_lh": "fs5_mask_lh.gii",
    "fs5_mask_rh": "fs5_mask_rh.gii",
    "fs5_mask": "fs5_mask_bh.gii",
}

files = {
    "connectivity_config": "connectivity_config.json",
    "dataset_description_template": "dataset_description_template.json",
}


def get_img(ds):
    """Get a standard image file used in nimlab as a nilearn nifti-like

    Parameters
    ----------
    ds : str
        Name of the img. Options are as follows:

        Volume Masks
        ---
        "MNI152_T1_1mm_brain"
        "MNI152_T1_1mm_brain_mask"
        "MNI152_T1_1mm_brain_mask_dil"
        "MNI152_T1_2mm_brain"
        "MNI152_T1_2mm_brain_mask"
        "MNI152_T1_2mm_brain_mask_dil"
        "MNI152_T1_2mm_brain_mask_dil1"
        "mni_icbm152"
        "222"

        Surface Masks
        ---
        "fs5_mask_lh"
        "fs5_mask_rh"
        "fs7_mask_lh"
        "fs7_mask_rh"
        "fs5_mask" (Gifti Format)

        Difference masks (voxels unique to each mask)
        ---
        "222_diff"
        "dil_diff"
        "dil1_diff"

    Returns:
        Niimg-like object: image
    """
    assert ds in datasets.keys(), "Unknown dataset specified"
    fname = datasets[ds]
    from . import data

    with pkg_resources.path(data, fname) as datafile:
        return nib.load(str(datafile))


def get_img_path(ds):
    """Get path for standard nimlab imgs

    Options are as follows:

        Volume Masks
        ---
        "MNI152_T1_1mm_brain"
        "MNI152_T1_1mm_brain_mask"
        "MNI152_T1_1mm_brain_mask_dil"
        "MNI152_T1_2mm_brain"
        "MNI152_T1_2mm_brain_mask"
        "MNI152_T1_2mm_brain_mask_dil"
        "MNI152_T1_2mm_brain_mask_dil1"
        "mni_icbm152"
        "222"

        Surface Masks
        ---
        "fs5_mask_lh"
        "fs5_mask_rh"
        "fs7_mask_lh"
        "fs7_mask_rh"
        "fs5_mask"

        Difference masks (voxels unique to each mask)
        ---
        "222_diff"
        "dil_diff"
        "dil1_diff"
    Args:
        ds (str): image name

    Returns:
        str: path to image
    """
    assert ds in datasets.keys(), "Unknown dataset specified"
    fname = datasets[ds]
    from . import data

    with pkg_resources.path(data, fname) as datafile:
        return str(datafile)


def get_filepath(f):
    """Get filepath for non-nifti data files.

    Options are:
    "connectivity_config": "connectivity_config.json",
    "dataset_description_template": "dataset_description_template.json"

    Args:
        f (str): Filename

    Returns:
        str: File path
    """
    assert f in files.keys(), "Unknown file specified"
    fname = files[f]
    from . import data

    with pkg_resources.path(data, fname) as datafile:
        return str(datafile)


def check_mask(f=""):
    """Check if mask name is in nimlab datasets or get list of available masks.

    Args:
        f (str): Mask name to check. If empty string, print list of available datasets and return False.

    """
    if f:
        if f in datasets.keys():
            return True
        else:
            print("Unknown dataset specified")
            return False
    else:
        print(datasets.keys())
        return False
