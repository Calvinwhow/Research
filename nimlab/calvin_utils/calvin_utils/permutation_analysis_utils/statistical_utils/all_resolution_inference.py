'''
1) AllResolutionInference
-------------------------
This module contains the AllResolutionInference class, which is designed to perform "All-Resolution Inference" on NIfTI files.

Example Usage:

    from AllResolutionInference import AllResolutionInference

    # initialize the class with the path to a NIfTI file
    ar_inference = AllResolutionInference('path/to/file.nii')

    # perform all-resolution inference and save the result
    ar_inference.save_all_resolution_inference('path/to/output.nii')

Notes:
- The class loads a provided NIfTI file which contains voxel-wise p-values, corrected using TFCE FWER, converts these p-values into z-scores, performs all-resolution inference on these z-scores, and saves the result back into a NIfTI file.
- The implementation assumes that the input and output files are in the NIfTI format (.nii or .nii.gz).
- The 'threshold' and 'alpha' parameters in all-resolution inference can be tuned according to the needs of the analysis.

References
----------
For the AllResolutionInference module:
Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ. All-Resolutions Inference for brain imaging. 
NeuroImage. 2018 Nov 1;181:786-796. doi: 10.1016/j.neuroimage.2018.07.060
'''

from nilearn.glm import cluster_level_inference
import nibabel as nib
import numpy as np
from scipy import stats

class AllResolutionsInference:
    '''
    Reference: 
    Python Implementation: https://nilearn.github.io/dev/auto_examples/05_glm_second_level/plot_proportion_activated_voxels.html
    Neuroimage Manuscript: Rosenblatt JD, Finos L, Weeda WD, Solari A, Goeman JJ. All-Resolutions Inference for brain imaging. Neuroimage. 2018 Nov 1;181:786-796. doi: 10.1016/j.neuroimage.2018.07.060
    '''
    def __init__(self, nii_file):
        """
        Initializes a new instance of the AllResolutionInference class.

        :param nii_file: The path to the .nii file.
        """
        self.nii_file = nii_file
        self.nii_img = self.load_nii(nii_file)
        self.nii_data = self.nii_img.get_fdata()

    def load_nii(self, nii_file):
        """
        Loads a .nii file and returns the corresponding image object.

        :param nii_file: The path to the .nii file.
        :return: A Nifti1Image object containing the image data.
        """
        try:
            img = nib.load(nii_file)
            return img
        except Exception as e:
            print(f"Failed to load file {nii_file}. Error: {str(e)}")
            return None

    def pvals_to_zscores(self):
        """
        Converts the p-values in the NIfTI data to z-scores and returns a new NIfTI image.

        :return: A Nifti1Image object containing the z-score data.
        """
        # Replace NaNs and infinities with large/small finite numbers
        finite_data = np.nan_to_num(self.nii_data, nan=np.NaN, posinf=0.0000000000001, neginf=0.0000000000001)

        # Convert p-values to z-scores
        zscores_data = stats.norm.ppf(1 - finite_data)

        # Create a new NIfTI image with the z-score data, keeping the original header
        zscores_img = nib.Nifti1Image(zscores_data, self.nii_img.affine, self.nii_img.header)

        return zscores_img

    def perform_all_resolution_inference(self):
        """
        Performs all-resolution inference on the NIfTI data.

        :return: A Nifti1Image object containing the result of the all-resolution inference.
        """
        zscores_img = self.pvals_to_zscores()
        proportion_true_discoveries_img = cluster_level_inference(zscores_img.get_fdata(), threshold=[1.64, 3, 5], alpha=0.05)
        return nib.Nifti1Image(proportion_true_discoveries_img, self.nii_img.affine, self.nii_img.header)

    def save_all_resolution_inference(self, output_file):
        """
        Performs all-resolution inference on the NIfTI data and saves the result to a new .nii file.

        :param output_file: The path where the output .nii file should be saved.
        """
        all_res_inf_img = self.perform_all_resolution_inference()
        nib.save(all_res_inf_img, output_file)
