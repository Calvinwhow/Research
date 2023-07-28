"""Tools for Surface Gifti files

"""

import os
import csv
import shutil
import numpy as np
import nibabel as nib
from nilearn import image
from scipy.io import savemat
from nimlab import datasets as nimds
from nimlab import functions as nimfs
from nimlab import configuration as config
from IPython.core.getipython import get_ipython
from IPython.display import display

try:
    from pathlib import Path
except:
    from pathlib2 import Path


class GiftiMasker:
    def __init__(self, mask_img):
        """
        Parameters
        ----------
        mask_img : Giimg-like object
            If str, path to binary gifti mask.
            If giftiimg, gifti image object.
            Assumes the gifti only contains a single 1-D data array.

        """
        if type(mask_img) == str:
            self.mask_img = nib.load(mask_img)
        else:
            self.mask_img = mask_img
        self.mask_data = self.mask_img.agg_data()
        (self.mask_idx,) = np.where(self.mask_data != 0)
        self.mask_shape = self.mask_data.shape
        self.mask_size = np.prod(self.mask_shape)

    def transform(self, giimg=None, weight=False):
        """Masks Gifti file into 1D array. Retypes to float32
        Parameters
        ----------
        giimg : Giimg-like object
            If string, consider it as a path to GIfTI image and call
            `nibabel.load()` on it. The '~' symbol is expanded to the user home
            folder. If it is an object, check if affine attribute is present,
            raise `TypeError` otherwise.
            If ndarray, consider it as a gifti data array.

        weight : bool, default False
            If True, transform the niimg with weighting. If False, transform the
            niimg without weighting.

        Returns
        -------
        region_signals : numpy.ndarray
            Masked Nifti file.

        """
        if type(giimg) == str:
            giimg_data = nib.load(giimg).agg_data().astype(np.float32)
        elif type(giimg) == np.ndarray:
            giimg_data = giimg
        else:
            giimg_data = giimg.agg_data().astype(np.float32)
        if weight:
            img = np.multiply(self.mask_data, giimg_data)
        else:
            img = giimg_data
        return np.take(img, self.mask_idx)

    def inverse_transform(self, flat_giimg=None):
        """Unmasks 1D array into 3D Gifti file. Retypes to float32.
        Parameters
        ----------
        flat_giimg : numpy.ndarray
            1D array to unmask.

        Returns
        -------
        giimg : nibabel.gifti.gifti.GiftiImage
            Unmasked Gifti.

        """
        new_img = np.zeros(self.mask_size, dtype=np.float32)
        new_img[self.mask_idx] = flat_giimg.astype(np.float32)
        return new_gifti_image(data=new_img)

    def mask(self, giimg=None):
        """Masks 3D Gifti file into Masked Gifti file.
        Parameters
        ----------
        giimg : Giimg-like object
            If string, consider it as a path to GIfTI image and call
            `nibabel.load()` on it. The '~' symbol is expanded to the user home
            folder. If it is an object, check if affine attribute is present,
            raise `TypeError` otherwise.

        Returns
        -------
        masked_giimg : nibabel.gifti.gifti.GiftiImage
            Masked Gifti image.

        """
        return self.inverse_transform(self.transform(giimg))


def new_gifti_image(data, intent=0, datatype=16, metadata=None):
    """NiBabel wrapper to generate a gifti image with data array and metadata.

    Parameters
    ----------
    data : ndarray
        1-D ndarray containing one hemisphere surface data.
    intent : int
        Intent code for Gifti File. Defaults to 0 (Intent = NONE).

        Available intent codes:
            NIFTI_INTENT_NONE - 0
            NIFTI_INTENT_CORREL - 2
            NIFTI_INTENT_TTEST - 3
            NIFTI_INTENT_ZSCORE - 5
            NIFTI_INTENT_PVAL - 22
            NIFTI_INTENT_LOGPVAL - 23
            NIFTI_INTENT_LOG10PVAL - 24
            NIFTI_INTENT_LABEL - 1002
            NIFTI_INTENT_POINTSET - 1008
            NIFTI_INTENT_TRIANGLE - 1009
            NIFTI_INTENT_TIME_SERIES - 2001
            NIFTI_INTENT_NODE_INDEX - 2002
            NIFTI_INTENT_SHAPE - 2005

        More intent codes can be found at: https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/group__NIFTI1__INTENT__CODES.html

    datatype : int
        Datatype for gifti image. Defaults to 16 (dtype = float32)

        Available datatypes:
            UINT8 - 2
            INT32 - 8
            FLOAT32 - 16

    metadata : nibabel.gifti.gifti.GiftiMetaData
        Metadata for gifti image.

    Returns
    -------
    nibabel.gifti.gifti.GiftiImage
        Gifti image with specified metadata and data array.

    """
    dtypes = {2: np.uint8, 8: np.int32, 16: np.float32}
    data = data.astype(dtypes[datatype])
    if metadata:
        metadata = nib.gifti.GiftiMetaData(metadata)
    gifti_data = nib.gifti.GiftiDataArray(data=data, intent=intent, datatype=datatype)
    gifti_img = nib.gifti.GiftiImage(meta=metadata, darrays=[gifti_data])
    return gifti_img


def legacy_to_gifti(pseudo_gifti, output_dir=""):
    """Convert legacy pseudo_gifti files to gifti.

    Parameters
    ----------
    pseudo_gifti : str
        Path to pseudo gifti file be converted. Should be a .nii/.nii.gz file.
    output_dir : str, optional
        Output directory, by default "". If not specified, converted gifti will be
        output to same directory as input file.

    """
    data = image.get_data(pseudo_gifti)
    image_size = np.prod(data.shape)
    data = np.reshape(data, (image_size,), order="F")
    if output_dir == "":
        output_dir = os.path.dirname(pseudo_gifti)
    filename = os.path.basename(pseudo_gifti).split(".nii")[0] + ".gii"
    gifti_img = new_gifti_image(data)
    gifti_img.to_filename(os.path.join(output_dir, filename))


def concat_hemispheres_to_csv(gifti_paths, output_dir="", mask=""):
    """Concatenate a list of giftis together into a csv to construct a data matrix for
       use with PALM. Assumes that input giftis contain a single 1-D data array with
       functional/statistical data for surface mesh vertices.

    Parameters
    ----------
    gifti_paths : str
        Path to a two-column CSV of paths to giftis where each row corresponds to a
        subject, the first column corresponds to the subject's Left Hemisphere ROI mask,
        and the second column corresponds to the subject's Right Hemisphere ROI mask.

            Format:
                /path/to/subject1/lh_roi.gii,/path/to/subject1/rh_roi.gii
                /path/to/subject2/lh_roi.gii,/path/to/subject2/rh_roi.gii
                /path/to/subject3/lh_roi.gii,/path/to/subject3/rh_roi.gii
                                                .
                                                .
                                                .

    output_dir : str, optional
        Output directory, by default "". If not specified, generated data matrix csv
        will be output to same directory as input csv.

    mask : str, optional
        Mask name (from nimlab.datasets). Defaults to no masking.
        Options:
            fs5_mask - fsaverage5 mask from Ryan Darby ADNI data

    """
    roi_files = []
    flist = open(gifti_paths)
    reader = csv.reader(flist, delimiter=",")
    for f in reader:
        roi_files.append(f)
    flist.close()
    if mask:
        masker = GiftiMasker(nimds.get_img(mask))
    if output_dir == "":
        output_dir = os.path.dirname(gifti_paths)
    subject_data = []
    for subject in roi_files:
        lh = nib.load(subject[0]).agg_data()
        rh = nib.load(subject[1]).agg_data()
        data = np.concatenate((lh, rh))
        if mask:
            data = masker.transform(data)
        subject_data.append(data)
    fname = os.path.join(output_dir, "data.csv")
    np.savetxt(fname, np.stack(subject_data), delimiter=",")


def concat_hemispheres_to_matfile(gifti_paths, output_dir=""):
    """Concatenate a list of giftis together into a csv to construct a matfile for
       use with CBIG RegistrationFusion. Assumes that input giftis contain a single 1-D
       data array with functional/statistical data for surface mesh vertices.

    Parameters
    ----------
    gifti_paths : str
        Path to a two-column CSV of paths to giftis where each row corresponds to a
        subject, the first column corresponds to the subject's Left Hemisphere ROI mask,
        and the second column corresponds to the subject's Right Hemisphere ROI mask.

            Format:
                /path/to/subject1/lh_roi.gii,/path/to/subject1/rh_roi.gii
                /path/to/subject2/lh_roi.gii,/path/to/subject2/rh_roi.gii
                /path/to/subject3/lh_roi.gii,/path/to/subject3/rh_roi.gii
                                                .
                                                .
                                                .

    output_dir : str, optional
        Output directory, by default "". If not specified, generated data matrix matfile
        will be output to same directory as input csv.

    """
    roi_files = []
    flist = open(gifti_paths)
    reader = csv.reader(flist, delimiter=",")
    for f in reader:
        roi_files.append(f)
    flist.close()
    if output_dir == "":
        output_dir = os.path.dirname(gifti_paths)
    out_fnames = []
    for subject in roi_files:
        subject_data = {}
        lh = nib.load(subject[0]).agg_data()
        rh = nib.load(subject[1]).agg_data()
        subject_data["lh_label"] = lh
        subject_data["rh_label"] = rh
        savemat(
            os.path.join(
                output_dir,
                nimfs.lcs(os.path.basename(subject[0]), os.path.basename(subject[1])),
            ),
            subject_data,
        )
        out_fnames.append(
            [
                os.path.join(
                    output_dir,
                    nimfs.lcs(
                        os.path.basename(subject[0]), os.path.basename(subject[1])
                    ),
                )
            ]
        )
    with open(os.path.join(output_dir, "matfiles.csv"), "w+", newline="") as file:
        write = csv.writer(file)
        write.writerows(out_fnames)


def threshold(gifti, threshold, direction="twosided", binarize=False, replace_val=0.0):
    """Threshold a gifti image

    Parameters
    ----------
    gifti : nibabel.gifti.gifti.GiftiImage
        Gifti image to be thresholded.
    threshold : float
        Threshold value.
    direction : str
        If "less", keep values less than or equal to the threshold value.
        If "greater", keep values greater than or equal to the threshold value.
        If "twosided", keep values less than -{threshold} and values greater than {threshold}. Defaults to "twosided".
    binarize : bool
        If True, binarize at threshold. Binarizes with selected direction. If twosided, set values greater than threshold to 1 and values less than -1*threshold to -1.
    replace_val : float
        Replace thresholded values with this value. Defaults to 0.

    Returns
    -------
    gifti : nibabel.gifti.gifti.GiftiImage
        Thresholded Gifti image.

    """
    data = gifti.agg_data().astype(np.float32)
    threshold = float(threshold)
    if direction == "less":
        if binarize:
            data[data > threshold] = 0
            data[data <= threshold] = 1
        else:
            data[data > threshold] = replace_val
    elif direction == "greater":
        if binarize:
            data[data < threshold] = 0
            data[data >= threshold] = 1
        else:
            data[data < threshold] = replace_val
    elif direction == "twosided":
        if binarize:
            data[data >= threshold] = 1
            data[data <= -1 * threshold] = -1
            data[(data < threshold) & (data > -1 * threshold)] = 0
        else:
            data[(data < threshold) & (data > -1 * threshold)] = replace_val
    return new_gifti_image(data=data)


def nan_to_num(gifti, nan=0, posinf=None, neginf=None):
    """Replace NaNs in Gifti image with a value using np.nan_to_num

    Parameters
    ----------
    gifti : nibabel.gifti.gifti.GiftiImage
        Gifti image to be modified.
    nan : float, optional
        Value to replace NaNs with, by default 0

    Returns
    -------
    new_gifti : nibabel.gifti.gifti.GiftiImage
        Modified Gifti image.

    Assumes NIFTIINTENT 0 and datatype FLOAT32

    """
    data = gifti.agg_data()
    data = np.nan_to_num(data, nan=nan, posinf=posinf, neginf=neginf)
    metadata = gifti.meta
    new_gifti = new_gifti_image(data, intent=0, datatype=16, metadata=metadata)
    return new_gifti


def copy_fsaverage(subject, subdir):
    """Copy fsaverage subject from FreeSurfer installed subjects_dir to a destination.

    Parameters
    ----------
    subject : str
        Name of fsaverage folder to copy. Options are:
        "fsaverage","fsaverage6","fsaverage5","fsaverage4","fsaverage4"
    subdir : str
        Path to destination where fsaverage folder should be copied to.
    """
    if subject not in [
        "fsaverage",
        "fsaverage6",
        "fsaverage5",
        "fsaverage4",
        "fsaverage4",
    ]:
        raise ValueError(f"{subject} is not a valid fsaverage subject!")
    else:
        config.verify_software(["freesurfer_path"])
        os.environ["FREESURFER_HOME"] = config.software["freesurfer_path"]
        fs_subdir = os.getenv("FREESURFER_HOME") + "/subjects"

        shutil.copytree(
            os.path.join(fs_subdir, subject),
            os.path.join(subdir, subject),
            dirs_exist_ok=True,
        )


def make_cortical_thickness_glm(fsgd, glmdir, subdir, tmpdir, fwhm=5):
    config.verify_software(["fsl_path", "freesurfer_path"])
    FSLDIR = config.software["fsl_path"]
    os.environ["FREESURFER_HOME"] = config.software["freesurfer_path"]
    os.environ["SUBJECTS_DIR"] = subdir
    os.environ["FSFAST_HOME"] = os.getenv("FREESURFER_HOME") + "/fsfast"
    os.environ["MNI_DIR"] = os.getenv("FREESURFER_HOME") + "/mni"
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

    script_path = str(
        Path(__file__).parents[0] / "scripts/calc_cort_thickness_glm_fs5.sh"
    )

    ipython.system(f"bash {script_path} {fsgd} {glmdir} {subdir} {tmpdir} {fwhm}")


def make_cortical_thickness_wmap(wmap_config, subdir, tmpdir, glmdir, outdir, fwhm=5):
    config.verify_software(["fsl_path", "freesurfer_path"])
    FSLDIR = config.software["fsl_path"]
    os.environ["FREESURFER_HOME"] = config.software["freesurfer_path"]
    os.environ["SUBJECTS_DIR"] = subdir
    os.environ["FSFAST_HOME"] = os.getenv("FREESURFER_HOME") + "/fsfast"
    os.environ["MNI_DIR"] = os.getenv("FREESURFER_HOME") + "/mni"
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

    script_path = str(
        Path(__file__).parents[0] / "scripts/calc_cort_thickness_wmap_fs5.sh"
    )

    ipython.system(
        f"bash {script_path} {wmap_config} {fwhm} {subdir} {tmpdir} {glmdir} {outdir}"
    )
