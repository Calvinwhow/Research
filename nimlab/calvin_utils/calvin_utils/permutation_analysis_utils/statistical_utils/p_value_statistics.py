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