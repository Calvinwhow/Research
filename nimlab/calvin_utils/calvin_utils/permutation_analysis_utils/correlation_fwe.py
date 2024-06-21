import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from typing import Tuple
from scipy.stats import rankdata
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti

class CalvinFWEMap():
    """
    This is a class to orchestrate a simple association between some Y variable of interest and voxelwise data (X variable).
    It will run FWE correction via the Maximum Statistic Correction method.

    Notes:
    ------
    - Running max_stat_method = pseudo_var_smooth will reduce the risk of the maximum stat
        being a numerically unstable result of a bad permutation.
    - SpearmanR implemented by Scipy requires looping over every voxel.
        However, it is more statistically powerful than the vectorized projection
        due to their more sophisticated ranking (which is too intensive for vectorization).
    - PearsonR does not suffer from the same issues as the SpearmanR. 
        However, you may then consider simply using the linear regression counterpart of this code. 
    - The vectorized implementation of Spearman correlation handles large matrices efficiently
        using broadcasting but may not handle ties as accurately as Scipy's implementation.
    
    Attributes:
    -----------
    neuroimaging_dataframe : pd.DataFrame
        DataFrame with neuroimaging data where each column represents a subject and each row represents a voxel.
    variable_dataframe : pd.DataFrame
        DataFrame where each column represents a subject and each row represents the variable to regress upon.
    method : str
        The association method to relate the voxelwise data to. Defaults to 'spearman'. Options: 'spearman' | 'pearson'.
    mask_path : str or None
        The path to the mask to use. If None, will threshold the voxelwise image itself by mask_threshold.
    mask_threshold : float
        The threshold to mask the neuroimaging data at.
    out_dir : str
        Output directory to save results.
    max_stat_method : str or None
        Method for maximum statistic correction. Options: None | 'pseudo_var_smooth' | 'var_smooth'.
    vectorize : bool
        Whether to use vectorized implementation for correlation calculation.

    Methods:
    --------
    sort_dataframes(voxel_df: pd.DataFrame, covariate_df: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]
        Sorts the rows of the voxelwise and covariate DataFrames to ensure they are identically organized.
    
    threshold_probabilities(df: pd.DataFrame, debug: bool=False) -> pd.Series
        Applies a threshold to mask raw voxelwise data.
    
    mask_dataframe(neuroimaging_df: pd.DataFrame) -> Tuple[pd.Index, pd.Series, pd.DataFrame]
        Applies a mask to the neuroimaging DataFrame based on nonzero voxels.
    
    unmask_dataframe(df: pd.DataFrame) -> pd.DataFrame
        Simple unmasking function to restore original dimensions.
    
    mask_by_p_values(results_df: pd.DataFrame, p_values_df: pd.DataFrame) -> pd.DataFrame
        Thresholds results by FWE corrected p-values.
    
    permute_covariates() -> pd.DataFrame
        Permutes the patient data by randomly assigning patient data to new patients.
    
    efficient_rankdata(arr: np.ndarray, axis: int = 0) -> np.ndarray
        Efficiently ranks data using numpy.
    
    run_spearman(X: np.array, Y: np.array, debug: bool = False) -> pd.DataFrame
        Calculates voxelwise relationship to Y variable with Spearman correlation.
    
    run_pearson(X: np.array, Y: np.array, debug: bool = False) -> pd.DataFrame
        Calculates voxelwise relationship to Y variable with Pearson correlation.
    
    correlation(permuted_variable_df: pd.DataFrame=None, debug: bool=False) -> pd.DataFrame
        Calculates voxelwise relationship to Y variable with the specified correlation method.
    
    var_smooth(df: pd.DataFrame)
        Takes the 95th percentile of the permuted data as the 'maximum stat' as a proxy for variance smoothed max stat.
    
    pseudo_var_smooth(df: pd.DataFrame) -> np.ndarray
        Takes the 95th percentile of the permuted data as the 'maximum stat' as a proxy for variance smoothed max stat.
    
    raw_max_stat(df: pd.DataFrame) -> float
        Returns the max statistic in the data.
    
    get_max_stat(df: pd.DataFrame) -> float
        Chooses the max stat method and returns the max stat.
    
    maximum_stat_fwe(n_permutations: int=100, debug: bool=False) -> list
        Performs maximum statistic Family-Wise Error (FWE) correction using permutation testing.
    
    p_value_calculation(uncorrected_df: pd.DataFrame, max_stat_dist: list, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]
        Calculates p-values for the uncorrected statistic values using the distribution of maximum statistics.
    
    save_single_nifti(nifti_df: pd.DataFrame, out_dir: str, name: str='generated_nifti', silent: bool=True)
        Saves NIFTI images to the specified directory.
    
    save_results(voxelwise_results: pd.DataFrame, unmasked_p_values: pd.DataFrame, voxelwise_results_fwe: pd.DataFrame)
        Saves the generated result files.
    
    run(n_permutations: int=100, debug: bool=False)
        Orchestration method to run the entire analysis and save the results.
    """
    def __init__(self, neuroimaging_dataframe: pd.DataFrame, variable_dataframe: pd.DataFrame, method: str='spearman', mask_path=None, mask_threshold: int=0.0, out_dir='', max_stat_method=None, vectorize=True):
        """
        Need to provide the dataframe dictionaries and dataframes of importance. 
        
        Args:
        - neuroimaging_dataframe (df): DF with neuroimaging data (voxelwise dataframe) column represents represents a subject,
                                        and each row represents a voxel.
        - variable_dataframe (pd.DataFrame): DataFrame where each column represents represents a subject,
                                        and each row represents the variable to regress upon. 
        - method (str): the association method to relate the voxelwise data to. Defaults to spearman correlation
                                        options: spearman | pearson
        - mask_path (str): the path to the mask you want to use. 
                                        If None, will threshold the voxelwise image itself by mask_threshold.
        - mask_threshold (int): The threshold to mask the neuroimaging data at.
        """
        self.method = method
        self.mask_path = mask_path
        self.mask_threshold = mask_threshold
        self.out_dir = out_dir
        self.max_stat_method = max_stat_method
        self.vectorize = vectorize
        neuroimaging_dataframe, self.variable_dataframe = self.sort_dataframes(covariate_df=variable_dataframe, voxel_df=neuroimaging_dataframe)
        self.original_mask, self.nonzero_mask, self.neuroimaging_dataframe = self.mask_dataframe(neuroimaging_dataframe)

    def sort_dataframes(self, voxel_df: pd.DataFrame, covariate_df: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Will sort the rows of the voxelwise DF and the covariate DF to make sure they are identically organized.
        Then will check that the columns are equivalent.
        """
        try:
            voxel_cols = voxel_df.columns.astype(int).values
            covariate_cols = covariate_df.columns.astype(int).values
            
            voxel_df.columns = voxel_cols
            covariate_df.columns = covariate_cols
            
            shared_columns = list(set(voxel_cols).intersection(set(covariate_cols)))
            
            if debug:
                # Print shared columns for debugging
                print("Shared Columns:", shared_columns)
                
                # Identify and print dropped columns
                dropped_voxel_cols = set(voxel_df.columns) - set(shared_columns)
                dropped_covariate_cols = set(covariate_df.columns) - set(shared_columns)
                
                if dropped_voxel_cols:
                    print("Dropped Voxel Columns:", dropped_voxel_cols)
                if dropped_covariate_cols:
                    print("Dropped Covariate Columns:", dropped_covariate_cols)
        except:
            # Force Columns to Match
            voxel_cols = set(voxel_df.columns.astype(str).sort_values().values)
            covariate_cols = set(covariate_df.columns.astype(str).sort_values().values)
            shared_columns = list(voxel_cols.intersection(covariate_cols))
        
        # Align dataframes to shared columns
        aligned_voxel_df = voxel_df.loc[:, shared_columns]
        aligned_covariate_df = covariate_df.loc[:, shared_columns]
        return aligned_voxel_df, aligned_covariate_df
    
    def threshold_probabilities(self, df: pd.DataFrame, debug=False) -> pd.Series:
        """
        Apply a threshold to mask raw voxelwise data. 
        Finds all voxels which are nonzero across all rows and create a mask from them. 
        
        Parameters:
        df (pd.DataFrame): DataFrame with voxelwise data.
        
        Returns:
        pd.Series: Mask of nonzero voxels.
        """
        if self.mask_path is not None: 
            mask_data = nib.load(self.mask_path).get_fdata().flatten()
            mask_data = pd.DataFrame(mask_data, index=df.index, columns=['mask_data'])
            if len(mask_data) != len(df):
                raise ValueError("Length of mask data does not match the length of the DataFrame. Resolution error suspected")
            mask_data_thr = mask_data.where(mask_data > self.mask_threshold, 0)
        else:
            mask_data_thr = df.where(df > self.mask_threshold, 0)

        mask_indices = mask_data_thr.sum(axis=1) > 0
        if debug:
            print(mask_indices.shape, np.max(mask_indices))
        return mask_indices
    
    def mask_dataframe(self, neuroimaging_df: pd.DataFrame):
        """
        Apply a mask to the neuroimaging DataFrame based on nonzero voxels.
        
        Parameters:
        neuroimaging_df (pd.DataFrame): DataFrame with neuroimaging data.
        
        Returns:
        pd.Index: Index of the whole DataFrame.
        pd.Series: Mask of nonzero voxels.
        pd.DataFrame: Masked neuroimaging DataFrame.
        """
        # Now you can use the function to apply a threshold to patient_df and control_df
        mask = self.threshold_probabilities(neuroimaging_df)
        
        original_mask = neuroimaging_df.index
        masked_neuroimaging_df = neuroimaging_df.loc[mask, :]
        return original_mask, mask, masked_neuroimaging_df
    
    def unmask_dataframe(self, df:pd.DataFrame):
        """
        Simple unmasking function.
        """
        # Initialize a new DF
        empty_mask = pd.DataFrame(index=self.original_mask, columns=['voxels'], data=0)

        # Insert data into the DF 
        empty_mask.loc[self.nonzero_mask, :] = df.values.reshape(-1, 1)
        return empty_mask
    
    def mask_by_p_values(self, results_df:pd.DataFrame, p_values_df:pd.DataFrame):
        """Simple function to perform the thresholding by FWE corrected p-values"""
        unmasked_df = results_df.copy()
        
        mask = p_values_df.where(p_values_df < 0.05, 0)
        mask = mask.sum(axis=1) == 0
        
        unmasked_df.loc[mask, :] = 0
        return unmasked_df
    
    def permute_covariates(self):
        """Permute the patient data by randomly assigning patient data (columnar data) to new patients (columns)"""
        return self.variable_dataframe.sample(frac=1, axis=1, random_state=None)
    
    def efficient_rankdata(self, arr, axis=0):
        """
        Efficiently rank data using numpy.
        
        Args:
            arr (np.ndarray): Array to be ranked.
            axis (int): Axis along which to rank the data.
        
        Returns:
            np.ndarray: Ranked data.
        """
        arr = np.asarray(arr)
        ranks = np.empty_like(arr, dtype=float)
        if axis == 0:
            sorter = np.argsort(arr, axis=axis)
            ranks[sorter, np.arange(arr.shape[1])] = np.arange(arr.shape[0])[:, np.newaxis] + 1
        else:
            sorter = np.argsort(arr, axis=axis)
            ranks[np.arange(arr.shape[0])[:, np.newaxis], sorter] = np.arange(arr.shape[1]) + 1
        return ranks
    
    def run_spearman(self, X: np.array, Y: np.array, debug: bool = False) -> pd.DataFrame:
        """
        Calculate voxelwise relationship to Y variable with Spearman correlation.

        Args:
            X (pd.DataFrame): DataFrame of independent variables (e.g., patients x variables).
            Y (pd.DataFrame): DataFrame of dependent variables (e.g., patients x voxels).
            debug (bool): If true, prints out summary metrics.

        Returns:
            pd.DataFrame: DataFrame of correlation coefficients.
        """
        if not self.vectorize:
            from scipy.stats import spearmanr
            #Initialize
            n_voxels = Y.shape[1]
            rho = np.zeros(n_voxels)
            
            # Iterate over each voxel
            for i in range(n_voxels):
                rho[i], _ = spearmanr(X, Y[:, i])
                
            if debug:
                print("X: ", X.shape, " Y: ", Y.shape)
                print('Spearman correlation matrix shape: ', rho.shape)

            r_df = pd.DataFrame(rho, index=np.arange(n_voxels), columns=['rho'])
        else:
            # Rank the data
            X_ranked = self.efficient_rankdata(X, axis=0)
            Y_ranked = self.efficient_rankdata(Y, axis=0)

            if X_ranked.shape[0] != Y_ranked.shape[0]:
                raise ValueError(f"The number of rows in X ({X_ranked.shape}) must match the number of rows in Y ({Y_ranked.shape}).")

            # Calculate differences in ranks
            D = np.square(X_ranked - Y_ranked)

            # Sum the squared differences across patients
            SIGMA_D = np.sum(D, axis=0)

            # Calculate Spearman correlation
            N = X_ranked.shape[0]
            rho = 1 - ( (6 * SIGMA_D) / (N * (N**2 - 1)) )
            if debug:
                print("X: ", X.shape, " Y: ", Y.shape, " X_ranked: ", X_ranked.shape, " Y_ranked: ", Y_ranked.shape)
                print("D: ", D.shape, " SIGMA_D: ", SIGMA_D.shape)
                print('Spearman correlation matrix shape: ', rho.shape)

            # Reshape R to DataFrame format
            r_df = pd.DataFrame(rho.T, index=self.neuroimaging_dataframe.index, columns=['rho'])
        return r_df
    
    def run_pearson(self, X: np.array, Y: np.array, debug: bool = False) -> pd.DataFrame:
        """Will run a vectorized or looping pearson r"""
        if not self.vectorize:
            # Initialize
            from scipy.stats import pearsonr
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

            n_voxels = Y.shape[1]
            rho = np.zeros(n_voxels)
            
            # Iterate over each voxel
            for i in range(n_voxels):
                rho[i], _ = pearsonr(X[:, 0], Y[:, i])
                
            if debug:
                print("X: ", X.shape, " Y: ", Y.shape)
                print('Spearman correlation matrix shape: ', rho.shape)
            r_df = pd.DataFrame(rho, index=np.arange(n_voxels), columns=['r'])
        else:
            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"The number of rows in X ({X.shape}) must match the number of rows in Y ({Y.shape}).")
            # Calculate Numerator
            X_BAR = X.mean(axis=0)[:, np.newaxis] # covariate average across patients
            Y_BAR = Y.mean(axis=0)[np.newaxis, :] # voxelwise average across patients
            X_C = X - X_BAR
            Y_C = Y - Y_BAR
            NUMERATOR = np.dot(X_C.T, Y_C) # Transposing to facilitate matmul
            
            # Calculate Denominator
            SST_X = np.sum( (X - X_BAR)**2, axis=0) # Sum across a variable across patients
            SST_Y = np.sum( (Y - Y_BAR)**2 , axis=0) # Sum across a voxel across patients
            DENOMINATOR = np.sqrt(SST_X * SST_Y)
    
            # Pearson
            r = NUMERATOR / DENOMINATOR
            
            if debug:
                print("X: ", X.shape, " Y: ", Y.shape, " X_BAR :", X_BAR.shape, " Y_BAR ", Y_BAR.shape, " Y_C: ", Y_C.shape, " X_C: ", X_C.shape, "NUMERATOR :", NUMERATOR.shape, "DENOMINATOR: ", DENOMINATOR.shape, "R: ", r.shape)
                print(np.max(X_C), np.max(Y_C), np.max(SST_X), np.max(SST_Y), DENOMINATOR)
            r_df = pd.DataFrame(r.T, index=self.neuroimaging_dataframe.index, columns=['r'])
        return r_df
            
    def correlation(self, permuted_variable_df: pd.DataFrame=None, debug: bool=False) -> pd.DataFrame:
        """
        Calculate voxelwise relationship to Y variable with correlation
        It is STRONGLY advised to set mask=True when running this.
        Args:
            debug (bool): if true, prints out summary metrics

        Returns:
            pd.DataFrame:
        """
        # Design matrix X for control group, outcomes Y for control group
        if permuted_variable_df is not None:
            X = permuted_variable_df.values.T
        else:
            X = self.variable_dataframe.values.T 
        Y = self.neuroimaging_dataframe.values.T
        
        if self.method == 'spearman':
            r_df = self.run_spearman(X, Y)
        elif self.method == 'pearson':
            r_df = self.run_pearson(X,Y)
        else:
            raise ValueError("Incorrect method specific. Options are 'spearman' | 'pearson'")
        
        if debug:
            print(X.shape, Y.shape, r_df.shape)
            print('Observed R max: ', np.max(r_df))
        return r_df
    
    def var_smooth(self, df):
        """Will take the 95th percentile of the permuted data as the 'maximum stat' as a proxy for variance smoothed max stat."""
        raise ValueError("Function not yet complete.")
    
    def pseudo_var_smooth(self, df):
        """Will take the 99.99th percentile of the permuted data as the 'maximum stat' as a fast proxy for variance smoothed max stat."""
        return np.array([[np.percentile(df, 99.99, axis=None)]])
    
    def raw_max_stat(self, df):
        """Will simply return the max statistic in the data"""
        return np.max(df)
    
    def get_max_stat(self,df):
        """Will choose the max stat methoda and return the max stat"""
        if self.max_stat_method is None:
            max_stat = self.raw_max_stat(df)
        elif self.max_stat_method == 'pseudo_var_smooth':
            max_stat = self.pseudo_var_smooth(df)
        elif self.max_stat_method == 'var_smooth':
            max_stat = self.var_smooth(df)
        else:
            raise ValueError("Invalid max_stat_method arg. Choose None | 'pseudo_var_smooth | 'var_smooth")
        return max_stat
    
    def maximum_stat_fwe(self, n_permutations=100, debug=False):
        """
        Perform maximum statistic Family-Wise Error (FWE) correction using permutation testing.

        This method calculates the maximum voxelwise R-squared values across multiple permutations
        of the covariates. It then uses these maximum statistics to correct for multiple comparisons,
        ensuring robust and conservative statistical inference.

        Args:
            n_permutations (int): Number of permutations to perform. Defaults to 100.

        Returns:
            list: A list of maximum R-squared values from each permutation.
        """
        max_stats = []
        for i in tqdm(range(0, n_permutations), desc='Permuting'):
            permuted_covariates = self.permute_covariates()
            permuted_df = self.correlation(permuted_covariates, debug=False)
            max_stat = self.get_max_stat(permuted_df)
            max_stats.append(max_stat)
            if debug:
                print("Permutation max stat: ", max_stat)
                print("Max stat shape: ", max_stat.shape)
        print('95th percentile of permuted statistic: ', np.percentile(max_stats, 95))
        if debug:
            print('5th percentile of permuted statistic: ', np.percentile(max_stats, 5))
        return max_stats
            
    def p_value_calculation(self, uncorrected_df, max_stat_dist, debug=False):
        """
        Calculate p-values for the uncorrected statistic values using the distribution of maximum statistics.

        Args:
            uncorrected_df (pd.DataFrame): DataFrame of uncorrected statistic values.
            max_stat_dist (list): Distribution of maximum statistic values from each permutation.

        Returns:
            np.ndarray: Array of p-values corresponding to the uncorrected statistic values.
        """
        # Calculate P-Values
        max_stat_dist = np.array(max_stat_dist)
        max_stat_dist = max_stat_dist[:, np.newaxis]
        if debug:
            print(max_stat_dist.shape, uncorrected_df.values.shape)
        p_values = np.mean(max_stat_dist >= uncorrected_df.values, axis=0)
        p_values_df = uncorrected_df.copy()
        p_values_df.loc[:,:] = p_values
        
        # Threshold by 95th Percentile of Max Status
        threshold = np.percentile(max_stat_dist, 95)
        corrected_df = uncorrected_df.where(uncorrected_df > threshold, 0)

        if debug:
            print(p_values_df.shape, f'\n Max in uncorrected DF: {np.max(uncorrected_df)} \n', f'Threshold: {threshold} \n', f'Max in corrected DF: {np.max(corrected_df)}')
        return p_values_df, corrected_df

    def save_single_nifti(self, nifti_df, out_dir, name='generated_nifti', silent=True):
        """Saves NIFTI images to directory."""
        preview = view_and_save_nifti(matrix=nifti_df,
                            out_dir=out_dir,
                            output_name=name,
                            silent=silent)
        return preview
        
    def save_results(self, voxelwise_results, unmasked_p_values, voxelwise_results_fwe):
        """
        Saves the generated files. 
        """
        self.uncorrected_img = self.save_single_nifti(nifti_df=voxelwise_results, out_dir=self.out_dir, name='uncorrected_results', silent=False)
        self.p_img = self.save_single_nifti(nifti_df=unmasked_p_values, out_dir=self.out_dir, name='p_values', silent=False)
        self.corrected_img = self.save_single_nifti(nifti_df=voxelwise_results_fwe, out_dir=self.out_dir, name='fwe_corrected_results', silent=False)

    def run(self, n_permutations=100, debug=False):
        """
        Orchestration method. 
        """
        #Can be abstracted to run the analysis of choice and return it and the p-values
        voxelwise_results = self.correlation(debug=debug)
        max_stat_dist = self.maximum_stat_fwe(n_permutations=n_permutations, debug=debug)
        p_values, voxelwise_results_fwe = self.p_value_calculation(voxelwise_results, max_stat_dist, debug=debug)
        # 
        voxelwise_results = self.unmask_dataframe(voxelwise_results)
        unmasked_p_values = self.unmask_dataframe(p_values)
        voxelwise_results_fwe = self.unmask_dataframe(voxelwise_results_fwe)
        self.save_results(voxelwise_results, unmasked_p_values, voxelwise_results_fwe)
        if debug:
            print(np.max(voxelwise_results), np.max(unmasked_p_values), np.max(voxelwise_results_fwe))
            print(voxelwise_results.shape, unmasked_p_values.shape, voxelwise_results_fwe.shape)