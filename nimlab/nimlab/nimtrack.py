import pathlib, json
from nilearn import image, datasets, maskers
from nimlab import datasets as nimds
from bids import layout
import numpy as np
import hashlib


def bids_sanitize(name):
    """Sanitize a string for use with BIDS formatting. BIDS disallows both underscores and hyphens,
    as those are reserved for delineating key-value pairs. This function replaces underscores with a
    lowercase u and hyphens with a lowercase h.

    Args:
        name (str): name to be sanitized
    """
    underscore_components = name.split("_")
    # replace underscores with lowercase u to be bids compliant
    unsnaked = "u".join(underscore_components)
    hyphen_components = unsnaked.split("-")
    unhyphened = "h".join(hyphen_components)
    return unhyphened


def infer_mask(img):
    """Infer the mask used for functional connectivity by examining voxels that are unique to that mask.
    Currently is able to detect 222, dil, dil1, and otherwise assumes it is the nilearn 152 mask.

    Args:
        img (niimg-like): Image to infer

    Returns:
        str: Name of the mask
    """
    diff_masks = {
        "222": datasets.get_img("222_diff"),
        "dil": datasets.get_img("dil_diff"),
        "dil1": datasets.get_img("dil1_diff"),
    }
    for m in diff_masks.keys():
        masker = maskers.NiftiMasker(diff_masks[m]).fit()
        if masker.transform(img).max() != 0:
            return m
    else:
        # The nilearn mask is the most restrictive, and has no voxels that are not present
        # in other masks
        return "nilearn"


def get_xnat_path_meta(path):
    """Get BIDS entities from an xnat-style path.
    The path must be relative to the dataset root. That is, the first directory in the
    path should be "subjects"

    Args:
        path (str): Image file path
    """
    if path[0] == "/":
        path = path[1:]
    if path[-1] == "/":
        path = path[:-1]

    parts = path.split("/")
    if parts[-1].split(".")[-1] != "gz" and parts[-1].split(".")[-2] != "nii":
        return None
    if len(parts) < 3:
        return None
    subject = parts[1]
    statistic = None
    # print(parts)
    sh_suffixes = {
        "_AvgR_Fz.nii.gz": "avgRFz",
        "_AvgR.nii.gz": "avgR",
        "_T.nii.gz": "t",
        "_VarR.nii.gz": "varR",
        "_AvgR_Fz_surf_lh.gii": "surfLhAvgRFz",
        "_AvgR_Fz_surf_rh.gii": "surfRhAvgRFz",
        "_AvgR_surf_rh.gii": "surfRhAvgR",
        "_AvgR_surf_lh.gii": "surfLhAvgR",
        "_T_surf_rh.gii": "surfRhAvgRFz",
        "_T_surf_lh.gii": "surfLhAvgRFz",
    }
    quick_suffixes = {
        "AvgR_Fz.nii.gz": "avgRFz",
        "AvgR.nii.gz": "avgR",
        "T.nii.gz": "t",
    }

    if parts[2] == "Lesion":
        datatype = "roi"
        suffix = "lesionMask"
        statistic = None
        connectome = None
    elif parts[2] == "Connectivity":
        datatype = "connectivity"
        # connectome.sh style
        if "func_seed" in path:
            stat_suffix = parts[-1].split("seed")[-1]
            statistic = sh_suffixes[stat_suffix]
        if "struc_seed" in path:
            statistic = "strucSeed"
        # connectome_quick style
        else:
            for k in quick_suffixes.keys():
                if k in path:
                    statistic = quick_suffixes[k]
                    break
        connectome = "yeo1k"
        suffix = None
    else:
        return None
    entities = {
        "subject": bids_sanitize(subject),
        "datatype": datatype,
        "statistic": statistic,
        "suffix": suffix,
        "connectome": connectome,
    }
    return entities


def convert_xnat_path(path, bids_root, prefix):
    """Convert an xnat-style path to a bids-style path

    Args:
        path (str): XNAT style path
        bids_root (str): root directory of BIDS dataset
        prefix (str): Path prefix (i.e. the part of the path before we get to the XNAT dataset directory)
    """
    if path[-1] == "/":
        path = path[:-1]
    if prefix[-1] == "/":
        prefix = prefix[:-1]
    if bids_root[-1] == "/":
        bids_root = bids_root[:-1]
    with open(nimds.get_filepath("connectivity_config")) as f:
        config = json.load(f)
    patterns = config["default_path_patterns"]
    dataset_name = path.split(prefix)[-1].split("/")[1]
    rel_path = path.split(dataset_name)[-1]
    entities = get_xnat_path_meta(rel_path)
    # print(entities)
    if entities is not None:
        entities_clean = {k: v for k, v in entities.items() if v is not None}
        converted = layout.writing.build_path(entities, patterns)
        converted_fullpath = bids_root + "/" + dataset_name + "/" + converted
        return converted_fullpath
    else:
        # case where unsorted
        return bids_root + "/" + dataset_name + "/unsorted/" + rel_path


def generate_sidecar(path, layout):
    """Generate json sidecar for nifti files. Currently supports AvgR, AvgR_Fz, T, and lesionMask
    images.

    Args:
        path (str): Path to image
        layout (BIDSLayout): BIDSLayout object for dataset
    """
    sidecar_entities = {}
    entities = layout.parse_file_entities(path)
    # Get lesion size
    if entities["datatype"] == "roi":
        img = image.load_img(path)
        sidecar_entities["roi_size"] = np.count_nonzero(~np.isnan(img.get_fdata()))
    # Get mask, but exclude the weird surface space format ones
    else:
        if entities["statistic"] not in [
            "surfLhAvgRFz",
            "surfRhAvgRFz",
            "surfRhAvgR",
            "surfLhAvgR",
            "surfRhAvgRFz",
            "surfRhAvgRFz",
            "surfLhAvgRFz",
        ]:
            sidecar_entities["mask"] = infer_mask(path)

    # Get hash
    hasher = hashlib.md5()
    with open(path, "rb") as afile:
        buf = afile.read()
        hasher.update(buf)
    sidecar_entities["md5"] = hasher.hexdigest()

    return sidecar_entities
