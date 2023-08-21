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
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import nibabel as nib
from glob import glob 
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
            max_val = np.max(df)
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

    def start_uncorrected_calculate(self):
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
    
    def load_nifti_data(self, nifti_path):
        """
        Load the data from a NIfTI file.

        Parameters:
        - nifti_path : str
            Path to the NIfTI file.

        Returns:
        - data : np.ndarray
            The data extracted from the NIfTI file.
        """
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        return data

    def calculate_p_values_nifti(self, thresholded_nifti_data, maxima_array):
        """
        Calculates the p-value for each voxel in the thresholded NIfTI data.

        Parameters:
        - thresholded_nifti_data : np.ndarray
            The thresholded data extracted from a NIfTI file.
        - maxima_array : np.array
            A numpy array containing the maximum value from each permuted results file.

        Returns:
        - p_values_nifti : nibabel.Nifti1Image
            A NIfTI image containing the p-values for each voxel.
        """
        
        # Create a boolean mask of non-NaN values
        mask_non_nan = ~np.isnan(thresholded_nifti_data)
        
        # For each voxel, calculate the proportion of maxima that are greater
        p_values_data = np.empty_like(thresholded_nifti_data)
        p_values_data[mask_non_nan] = (maxima_array[:, None, None, None] > thresholded_nifti_data[mask_non_nan]).mean(axis=0)
        
        # Set NaN values for masked-out regions
        p_values_data[~mask_non_nan] = np.NaN
        
        p_values_nifti = nib.Nifti1Image(p_values_data, np.eye(4))
        return p_values_nifti

    def extract_without_multiprocess(self, directory, basename):
        """
        Extract the maximum value from each CSV file in the specified directory with the given basename.
        
        Parameters:
        - directory (str): The path to the directory containing the CSV files.
        - basename (str): The basename for the CSV files to be processed.
        
        Returns:
        - List of maximum values from each CSV file.
        """
        search_pattern = f"{directory}/{basename}*.csv"
        csv_files = glob(search_pattern)
        max_values = []

        for csv_file in tqdm(csv_files):
            data = pd.read_csv(csv_file, header=0)
            max_value = np.max(data)
            max_values.append(max_value)
        
        return np.array(max_values)

    def threshold_nifti_data(self, nifti_data, threshold):
        """
        Threshold the provided NIfTI data based on the given threshold value.
        All values below the threshold are set to NaN.
        
        Parameters:
        - nifti_data (np.ndarray): The 3D numpy array representing the NIfTI data.
        - threshold (float): The threshold value.
        
        Returns:
        - np.ndarray: The thresholded NIfTI data.
        """
        thresholded_data = nifti_data.copy()
        thresholded_data[thresholded_data < threshold] = np.NaN
        return thresholded_data

    def fwe_calculate(self, directory=None, basename=None, nifti_path=None, use_nifti=False, multiprocess=True):
        # Extract maxima
        if multiprocess:
            maxima_array = self.extract_maxima()
        else:
            maxima_array = self.extract_without_multiprocess(directory, basename)
        
        # Calculate 95th percentile
        percentile_95 = self.calculate_percentile(maxima_array)

        if use_nifti:
            # Load the NIfTI file and extract its data and affine
            nifti_img = nib.load(nifti_path)
            nifti_data = nifti_img.get_fdata()
            
            # Threshold the NIfTI data based on the 95th percentile
            thresholded_data = self.threshold_nifti_data(nifti_data, percentile_95)
            
            # Calculate voxelwise p-values for the NIfTI data
            p_values_data = self.calculate_p_values_nifti(thresholded_data, maxima_array)
            p_values_nifti = nib.Nifti1Image(p_values_data, nifti_img.affine)
            
            return thresholded_data, p_values_nifti
        
        else:
            # Threshold the observed distribution
            thresholded_data = self.threshold_distribution(self.observed_distribution, percentile_95)
            
            # Calculate p-values
            p_values = self.calculate_p_values(thresholded_data, maxima_array)
            
            return thresholded_data, p_values
    
    
class VarianceSmoothedPValueCalculator(PermutationPValueCalculator):
    """
    A class used to calculate variance-smoothed maximum-statistic FWE adjusted p-values.
    It inherits from the PermutationPValueCalculator class.
    
    Theory:
    -------
    The variance-smoothed maximum statistic method introduces a balance between the raw 
    variances of the test statistics and the average variance across all test statistics.
    The purpose is to stabilize variance estimates, especially in cases where the number of 
    tests or comparisons is large, potentially leading to unstable variance estimates for 
    individual tests.
    
    Mathematics:
    ------------
    For each test statistic (or voxel in imaging data), the variance is smoothed as:
    
    smoothed_variance_i = (1 - λ) * variance_i + λ * mean_variance
    
    where:
    - variance_i is the raw variance of the i-th test statistic.
    - mean_variance is the average variance across all test statistics.
    - λ (lambda) is a smoothing parameter between 0 and 1. A value of 0.5 indicates equal 
      weighting between the raw variance and the average variance.
      
    The test statistics are then standardized using the smoothed variances:
    
    standardized_statistic_i = test_statistic_i / sqrt(smoothed_variance_i)
    
    The maximum statistic method is applied on these standardized statistics for FWE correction.
    ------------
    
    The variance-smoothing procedure involves the following steps:
    1. For each voxel, the variance across the permuted distributions is calculated.
    2. A weighted average of the individual voxel variances and the average variance across all voxels is computed.
    3. This weighted variance is then used to standardize the data, i.e., divide each voxel value by the square root of its smoothed variance.
    4. The maxima of the smoothed permuted distributions are then identified.
    5. For each voxel in the smoothed observed distribution, its percentile in the smoothed permuted distribution is calculated.
    6. The p-value for each voxel is then determined as (1 - percentile).
    
    The method relies on the concept that smoothing the variance provides more robustness in FWE correction 
    by reducing the influence of outlier variances.
    
    Attributes
    ----------
    lambda_value : float
        A value between 0 and 1 that determines the degree of smoothing. A lambda value of 0 means no smoothing, 
        while a value of 1 means full smoothing towards the average variance across voxels.

    Methods
    -------
    variance_smooth_data(distribution)
        Applies variance smoothing and standardization to the provided distribution.
    
    calculate_smoothed_percentile(value, smoothed_distribution)
        Calculates the percentile of the given value in the smoothed distribution.
    
    run()
        Performs variance-smoothed FWE correction on the observed distribution using permuted distributions.
    """

    def __init__(self, observed_distribution, input_files, lambda_value=0.5):
        """
        Constructs all the necessary attributes for the VarianceSmoothedPValueCalculator object.
        
        Parameters
        ----------
        observed_distribution : pandas.DataFrame
            A dataframe representing the observed distribution.
        input_files : list
            A list of paths to files that represent the permuted results.
        lambda_value : float, optional
            The lambda value for variance smoothing (default is 0.5).
        """
        super().__init__(observed_distribution, input_files)
        self.lambda_value = lambda_value

    def variance_smooth_data(self, distribution):
        """
        Apply variance smoothing and standardization to the provided distribution.
        
        Parameters
        ----------
        distribution : pandas.DataFrame
            The distribution to be smoothed.
        
        Returns
        -------
        pandas.DataFrame
            The variance-smoothed and standardized distribution.
        """
        individual_variances = distribution.var()
        avg_variance = individual_variances.mean()
        smoothed_variances = (1 - self.lambda_value) * individual_variances + self.lambda_value * avg_variance
        standardized_data = distribution / np.sqrt(smoothed_variances)
        return standardized_data

    def calculate_smoothed_percentile(self, value, smoothed_distribution):
        """
        Calculate the percentile of the given value in the smoothed distribution.
        
        Parameters
        ----------
        value : float
            The value for which the percentile needs to be determined.
        smoothed_distribution : pandas.Series
            The smoothed distribution against which the percentile is calculated.
        
        Returns
        -------
        float
            The calculated percentile.
        """
        return (smoothed_distribution < value).mean()

    def _extract_values_from_file(self, file):
        """
        Helper function to extract all values from the given file using numpy for efficiency.
        If numpy fails, will use Pandas.
        """
        try:
            values = np.genfromtxt(file, delimiter=',').flatten()
        except:
            df = pd.read_csv(file)
            values = df.values.flatten()
        return values

    def run(self):
        """
        Perform variance-smoothed FWE correction on the observed distribution using permuted distributions.
        
        The process involves:
        1. Extracting the permuted distribution values from the input files.
        2. Applying variance-smoothing to the permuted distributions.
        3. Identifying the maxima of the smoothed permuted distributions.
        4. Smoothing the observed distribution.
        5. For each voxel in the smoothed observed distribution, calculating its percentile in the smoothed 
           permuted distribution maxima.
        6. Computing p-values for each voxel as (1 - percentile).
        
        Returns
        -------
        pandas.DataFrame
            A DataFrame with p-values for each voxel.
        """
        # Step 1: Extract the permuted distribution from a group of files
        permuted_distributions = [self._extract_values_from_file(file) for file in self.input_files]
        
        # Convert list of arrays to a DataFrame
        permuted_df = pd.DataFrame(np.array(permuted_distributions).T)

        # Step 2: Variance-smooth the permuted distribution
        smoothed_permuted_distributions = self.variance_smooth_data(permuted_df)

        # Step 3: Identify the maxima of the smoothed permuted distributions
        smoothed_maxima = smoothed_permuted_d
