"""
NeuroImage Analysis Modules
===========================

This library provides a set of modules for performing neuroimaging analysis.

Modules
-------

1) PermutationPValueCalculator
------------------------------
This module contains the PermutationPValueCalculator class, used to calculate p-values and thresholded 
distributions from observed distributions and permuted results files.

Example Usage:

    import pandas as pd
    from PermutationPValueCalculator import PermutationPValueCalculator

    observed_distribution = pd.read_csv('observed_distribution.csv')
    input_files = ['file1.csv', 'file2.csv', ..., 'file10000.csv']

    calculator = PermutationPValueCalculator(observed_distribution, input_files)
    thresholded_distribution, p_values = calculator.calculate()

Notes:
- The permuted results files are assumed to be CSV files that can be loaded into pandas DataFrames.
- The performance of the maxima extraction step can be improved by increasing the number of parallel jobs specified by n_jobs in the extract_maxima method.
- The p-values are calculated as the proportion of maxima that are greater than the sum of the voxel values in the thresholded distribution.
- The calculate_percentile method currently calculates the 95th percentile. Different percentiles can be calculated by specifying the percentile parameter when calling the method.

2) TFCalculator
---------------
This module provides the TFCalculator class for performing Threshold-Free Cluster Enhancement (TFCE) on 3D Nifti images.
This can also accept a 4D nifti, where the 4th dimension represents permutations. 

Example Usage:

    from TFCalculator import TFCalculator

    calculator = TFCalculator(E=0.5, H=2, dh=0.1) # These are default parameters
    calculator.process_folder('/path/to/nii/files')

Notes:
- The TFCalculator class uses the standard formula for TFCE and applies it to each voxel in the 3D image. The image data should represent a 3D map of test statistics (such as T-statistics).
- In this implementation, the E and H parameters are customizable, as well as the step size for the height increment (dh).
- The class assumes the input files are .nii format and outputs the results as new .nii files with '_tfce' appended to the original filenames.

3) AllResolutionInference
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
"""


import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label
from joblib import Parallel, delayed
from scipy.stats import norm, percentileofscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import maybe_unwrap_results
from joblib import Parallel, delayed
from scipy import stats
import nibabel as nib
from nilearn.glm import cluster_level_inference
import numpy as np


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


class TFCalculator:
    """
    This class is responsible for calculating Threshold-Free Cluster Enhancement (TFCE) for Nifti files.
    
    Reference: 
    TFCE Neuroimage paper: 
    White Paper: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf 
    Cython comparison: https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/tmanalysis/STEP_1_voxel_tfce_multiple_regression.py 
    MatLab comparison: https://github.com/markallenthornton/MatlabTFCE
    
    To Enable Inference Upon Voxels Within the Clusters, Implement This: 
        All-Resoutions Inference for Brain Imaging: https://www.sciencedirect.com/science/article/abs/pii/S105381191830675X?via%3Dihub
    """
    def __init__(self, E=0.5, H=2, dh=0.1):
        """
        Initializes a new instance of the TFCalculator class.
        
        :param E: The E parameter of the TFCE formula. Default is 0.5.
        :param H: The H parameter of the TFCE formula. Default is 2.
        :param dh: The step size for the height increment in the TFCE formula. Default is 0.1.
        """
        self.E = E
        self.H = H
        self.dh = dh
        
    def tfce_transform_indexed(self, stats):
        """
        Performs the TFCE transformation on a given map of abitrary statistics.

        :param tstats: A 3D numpy array containing the arbitrary statistics.
        :return: A 3D numpy array containing the TFCE values.
        """
        tfce_map = np.zeros_like(stats)
        cluster_labels, num_clusters = label(stats > 0)

        for cluster_id in range(1, num_clusters + 1):
            cluster_map = cluster_labels == cluster_id
            max_height = stats[cluster_map].max()
            height_steps = np.arange(self.dh, max_height + self.dh, self.dh)

            # Get all clusters at each height in one go
            clusters_at_heights = (stats[..., None] >= height_steps) & cluster_map[..., None]

            # Calculate extents for all clusters at all heights in one go
            extents = clusters_at_heights.sum(axis=(0, 1, 2))

            # Add contributions from all heights for this cluster in one go
            tfce_map[cluster_map] += np.sum((extents ** self.E * height_steps ** self.H * self.dh) * clusters_at_heights, axis=-1)

        return tfce_map

    def tfce_transform_looped(self, tstats):
        """
        Performs the TFCE transformation on a given map of T-statistics.
        
        :param tstats: A 3D numpy array containing the T-statistics.
        :return: A 3D numpy array containing the TFCE values.
        """
        tfce_map = np.zeros_like(tstats)
        cluster_labels, num_clusters = label(tstats > 0)

        for cluster_id in range(1, num_clusters+1):
            cluster_map = cluster_labels == cluster_id
            max_height = tstats[cluster_map].max()
            height_steps = np.arange(self.dh, max_height + self.dh, self.dh)

            for h in height_steps:
                cluster_at_height = cluster_map & (tstats >= h)
                extent = cluster_at_height.sum()
                tfce_map[cluster_at_height] += extent ** self.E * h ** self.H * self.dh

        return tfce_map

    def compute_tfce(self, nii_file, no_save=False):
        """
        Loads a .nii file, applies the TFCE transform, and optionally saves the result to a new .nii file.
        
        This method loads a .nii file, computes the TFCE of the 3D image in the file, and either
        saves the TFCE-transformed image to a new .nii file or returns the maximum TFCE value 
        without saving the file.
        
        :param nii_file: The path to the .nii file.
        :param no_save: A boolean indicating whether to save the TFCE-transformed image. If True,
                        the method returns the maximum TFCE value and does not save the image. 
                        If False, the method saves the TFCE-transformed image and does not return 
                        the maximum TFCE value. Default is False.
        :return: If no_save is True, returns the maximum TFCE value. Otherwise, does not return a value.
        """
        try:
            img = nib.load(nii_file)
            data = img.get_fdata()
            tfce_data = self.tfce_transform_indexed(data)  # Or use self.tfce_transform_looped(data) as per your needs
            tfce_img = nib.Nifti1Image(tfce_data, img.affine)
            tfce_file = os.path.splitext(nii_file)[0] + '_tfce.nii.gz'
            if no_save:
                return np.max(tfce_data)
            else:
                nib.save(tfce_img, tfce_file)
        except Exception as e:
            print(f"Failed to process file {nii_file}. Error: {str(e)}")

    def load_nii(self, nii_file):
            """
            Loads a .nii file and returns the corresponding numpy array.
            
            :param nii_file: The path to the .nii file.
            :return: A numpy array containing the image data.
            """
            try:
                img = nib.load(nii_file)
                return img.get_fdata()
            except Exception as e:
                print(f"Failed to load file {nii_file}. Error: {str(e)}")
                return None

    def load_and_process(self, nii_file):
        '''
        Takes nifti file path, loads it and processes it.
        
        :param nii_file: The path to the .nii file.
        :return: A numpy array containing the image data.
        '''
        data = self.load_nii(nii_file)
        self.compute_tfce(data)

        
    def process_folder(self, folder):
        """
        Processes all .nii files in a given folder.
        
        :param folder: The path to the folder containing the .nii files.
        """
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            return

        nii_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nii')]

        if not nii_files:
            print(f"No .nii files found in the folder {folder}.")
            return

        Parallel(n_jobs=-1)(delayed(self.load_and_process)(nii_file) for nii_file in nii_files)

    def extract_max_values(self, file_path):
        """
        Extracts the maximum Threshold-Free Cluster Enhancement (TFCE) values from each 3D image 
        in a 4D statistical image and returns them in an array.
        
        For each 3D image in the 4D statistical image, this method computes the TFCE and extracts 
        the maximum TFCE value. It does not save the TFCE-transformed images.
        
        :argument: filepath: Path to the file containing the 4D statistical image
        
        :return: A 1D numpy array containing the maximum TFCE value from each 3D image.
        """
        data = self.load_nii(file_path)
        if data is not None:
            max_values = np.full(self.data.shape[-1], np.nan)
            for i in range(self.data.shape[-1]):
                max_values[i] = self.compute_tfce(self.data[..., i], no_save=True)
            return max_values
        else:
            print(f"Failed to progress. Nifti generated array=None.")
            return None


class PermutationPValueCalculator:
    """
    A class used to calculate p-values and thresholded distributions from observed 
    distributions and permuted results files.
    Reference:
    FWER Methods: https://journals-sagepub-com.uml.idm.oclc.org/doi/epdf/10.1191/0962280203sm341ra
    Comparison of Multiple Correciton Methods: https://www.mdpi.com/2076-3425/9/8/198 | https://www.sciencedirect.com/science/article/pii/S1053811920302470 
    ...

    Attributes
    ----------
    observed_distribution : pandas.DataFrame
        a dataframe representing the observed distribution
    input_files : list
        a list of paths to files that represent the permuted results

    Methods
    -------
    _extract_max_from_file(file)
        Extracts maximum value from the given file.

    extract_maxima()
        Reads all the input files in parallel and extracts the maximum value from each.

    calculate_percentile(maxima_array, percentile=95)
        Calculates the percentile of the maxima_array.

    threshold_distribution(observed_distribution, threshold)
        Thresholds the observed_distribution such that only values over the given threshold are nonzero.

    calculate_p_values(thresholded_distribution, maxima_array)
        Calculates the p-value at all surviving voxels.

    calculate()
        Encapsulates the entire calculation flow, returning the thresholded input image and p-values image.
    """

    def __init__(self, observed_distribution, input_files):
        """
        Constructs all the necessary attributes for the PermutationPValueCalculator object.

        Parameters
        ----------
            observed_distribution : pandas.DataFrame
                a dataframe representing the observed distribution
            input_files : list
                a list of paths to files that represent the permuted results
        """
        self.observed_distribution = observed_distribution
        self.input_files = input_files

    def _extract_max_from_file(self, file):
        """
        Helper function to extract maximum value from the given file using numpy for efficiency.
        If numpy fails, will use Pandas.

        Parameters
        ----------
            file : str
                path to a permuted results file

        Returns
        -------
            max_val : float
                the maximum value in the file
        """
        try:
            data = np.genfromtxt(file, delimiter=',')
            max_val = data.max()
        except:
            df = pd.read_csv(file)
            max_val = df.max().max()
        return max_val

    def extract_maxima(self):
        """
        Reads all the input files in parallel and extracts the maximum value from each.

        Returns
        -------
            maxima_array : np.array
                a numpy array containing the maximum value from each permuted results file
        """
        # Use parallel processing for speed up
        maxima = Parallel(n_jobs=-1)(delayed(self._extract_max_from_file)(file) for file in self.input_files)
        return np.array(maxima)

    def calculate_percentile(self, maxima_array, percentile=95):
        """
        Calculates the percentile of the maxima_array.

        Parameters
        ----------
            maxima_array : np.array
                a numpy array containing the maximum value from each permuted results file
            percentile : int, optional
                percentile to calculate (default is 95)

        Returns
        -------
            perc : float
                the calculated percentile value
        """
        return np.percentile(maxima_array, percentile)

    def threshold_distribution(self, observed_distribution, threshold):
        """
        Thresholds the observed_distribution such that only values over the given threshold are nonzero.

        Parameters
        ----------
            observed_distribution : pandas.DataFrame
                a dataframe representing the observed distribution
            threshold : float
                the value to use for thresholding

        Returns
        -------
            thresholded_distribution : pandas.DataFrame
                the thresholded observed distribution
        """
        return observed_distribution.where(observed_distribution > threshold)

    def calculate_p_values(self, thresholded_distribution, maxima_array):
        """
        Calculates the p-value at all surviving voxels.

        Parameters
        ----------
            thresholded_distribution : pandas.DataFrame
                the thresholded observed distribution
            maxima_array : np.array
                a numpy array containing the maximum value from each permuted results file

        Returns
        -------
            p_values : pandas.DataFrame
                a dataframe containing the p-values at all surviving voxels
        """
        p_values = thresholded_distribution.copy()
        for voxel in p_values.columns:
            p_values[voxel] = (maxima_array > thresholded_distribution[voxel].sum()).mean()
        return p_values

    def fwe_calculate(self):
        """
        Encapsulates the entire calculation flow, returning the thresholded input image and p-values image.

        Returns
        -------
            thresholded_distribution : pandas.DataFrame
                the thresholded observed distribution
            p_values : pandas.DataFrame
                a dataframe containing the p-values at all surviving voxels
        """
        # Extract maxima
        maxima_array = self.extract_maxima()
        
        # Calculate 95th percentile
        percentile_95 = self.calculate_percentile(maxima_array)
        
        # Threshold the observed distribution
        thresholded_distribution = self.threshold_distribution(self.observed_distribution, percentile_95)
        
        # Calculate p-values
        p_values = self.calculate_p_values(thresholded_distribution, maxima_array)

        return thresholded_distribution, p_values

    def _extract_values_from_file(self, file):
        """
        Helper function to extract all values from the given file using numpy for efficiency.
        If numpy fails, will use Pandas.

        Parameters
        ----------
            file : str
                path to a permuted results file

        Returns
        -------
            values : np.array
                the values in the file
        """
        try:
            values = np.genfromtxt(file, delimiter=',').flatten()
        except:
            df = pd.read_csv(file)
            values = df.values.flatten()
        return values

    def calculate_uncorrected_p_values(self):
        """
        Calculates the uncorrected p-value at each voxel.

        Returns
        -------
            uncorrected_p_values : pandas.DataFrame
                a dataframe containing the uncorrected p-values at all voxels
        """
        uncorrected_p_values = self.observed_distribution.copy()
        for file in self.input_files:
            perm_values = self._extract_values_from_file(file)
            for voxel in uncorrected_p_values.columns:
                uncorrected_p_values[voxel] = (perm_values > uncorrected_p_values[voxel]).mean()
        return uncorrected_p_values

    def uncorrected_calculate(self):
        """
        Encapsulates the entire uncorrected calculation flow, returning the observed distribution and uncorrected p-values.

        Returns
        -------
            observed_distribution : pandas.DataFrame
                the observed distribution
            uncorrected_p_values : pandas.DataFrame
                a dataframe containing the uncorrected p-values at all voxels
        """
        # Calculate uncorrected p-values
        uncorrected_p_values = self.calculate_uncorrected_p_values()

        return self.observed_distribution, uncorrected_p_values