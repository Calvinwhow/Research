import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import zscore
from scipy.stats import t, norm
from nilearn import image, plotting 
from nimlab import datasets as nimds
from nibabel.affines import apply_affine

import numpy as np
import pandas as pd

def remove_unstable_values(df, posinf_val, neginf_val):
    """
    Removes unstable values from a DataFrame by setting:
    - NaNs to 0
    - Positive infinity to 'posinf_val'
    - Negative infinity to 'neginf_val'

    Parameters:
    - df: The input DataFrame from which to remove unstable values.
    - posinf_val: The value to use to replace positive infinity.
    - neginf_val: The value to use to replace negative infinity.

    Returns:
    - A DataFrame with unstable values replaced as specified.
    
    Example usage:
    # Assuming 'matrix_df' is your original DataFrame
    posinf_replacement_value = 1e6  # Example value for positive infinity replacement
    neginf_replacement_value = -1e6  # Example value for negative infinity replacement

    cleaned_df = remove_unstable_values(matrix_df, posinf_replacement_value, neginf_replacement_value)
    print(cleaned_df)
    """
    # Replace NaNs with 0
    df_cleaned = df.fillna(0)

    # Replace positive and negative infinity
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], [posinf_val, neginf_val])

    return df_cleaned

def join_dataframes(matrix_df1, matrix_df2):
    """
    Joins two dataframes side by side and returns the merged dataframe.
    
    Parameters:
    - matrix_df1 (DataFrame): The first dataframe to join.
    - matrix_df2 (DataFrame): The second dataframe to join.
    
    Returns:
    - DataFrame: The merged dataframe created by joining matrix_df1 and matrix_df2 side by side.
    """
    
    # Reset indexes of the input dataframes
    matrix_df1.reset_index()
    matrix_df2.reset_index()
    
    # Print lengths of the dataframes
    print('df1 len: ', len(matrix_df1), ' matrix_df2 len: ', len(matrix_df2))
    
    # Concatenate the dataframes side by side
    merged_df = pd.concat([matrix_df1, matrix_df2], axis=1, ignore_index=False)
    
    # Print the number of non-zero elements in the last column
    print('Nonzero values in last column: ', np.count_nonzero(merged_df.iloc[:,-1]))
    try:
        merged_df.pop('index')
    except:
        pass
    return merged_df

def handle_special_values(df):
    '''
    A quick and easy way of handling nans and inifinities in your data without significantly biasing the distribution.
    '''
    max_val = df.replace([np.inf, -np.inf], np.nan).max().max()
    min_val = df.replace([np.inf, -np.inf], np.nan).min().min()
    df = pd.DataFrame(np.nan_to_num(df.to_numpy(), nan=0.0, posinf=max_val, neginf=min_val), columns=df.columns)
    return df

def import_nifti_to_numpy_array(filepath):
    '''
    Does what it says. Just provide the absolute filepath.
    Args:
        filepath: absolute path to the file to import
    Returns:
        nifti_data: nifti_data as a numpy array
    '''
    try:
        # Load the NIfTI image using nilearn
        nifti_img = image.load_img(filepath)

        # Convert the NIfTI image to a NumPy array
        nifti_data = nifti_img.get_fdata()

        # Return the NIfTI image
        return nifti_data
    except Exception as e:
        print("Error:", e)
        return None
    
def view_nifti_html(img):
    html_image = plotting.view_img(img, cut_coords=(0,0,0), black_bg=False, opacity=.75, cmap='ocean_hot')
    return html_image

def threshold_matrix(matrix, threshold=95, method='percentile', direction='keep_above', output='zero', mask_mode=False):
    """
    Thresholds the matrix based on the provided criteria.
    
    Parameters:
        - matrix (np.array): The input matrix.
        - threshold (float or tuple): The threshold value(s). If tuple, interpreted as range.
        - method (str): How to determine the threshold. 'raw', 'probability', or 'percentile'.
        - direction (str): 'keep_above', 'keep_below', 'keep_between', or 'exclude_between' (if threshold is a tuple).
        - output (str): What to set the thresholded values to. 'zero' or 'nan'.
        - mask_mode (bool): If True, sets all non-masked values to 1.
    
    Returns:
        - np.array: Thresholded matrix.
    """
    print('--------------------------------Performing threshold--------------------------------')
    
    # Based on method, set the threshold value
    if isinstance(threshold, tuple):
        if method == 'probability':
            lower_threshold = st.norm.ppf(threshold[0])
            upper_threshold = st.norm.ppf(threshold[1])
        elif method == 'percentile':
            lower_threshold = np.percentile(matrix, threshold[0])
            upper_threshold = np.percentile(matrix, threshold[1])
        else:  # raw
            lower_threshold, upper_threshold = threshold
    else:
        if method == 'probability':
            threshold_value = st.norm.ppf(threshold)
        elif method == 'percentile':
            threshold_value = np.percentile(matrix, threshold)
            print(threshold_value)
        else:  # raw
            threshold_value = threshold
    
    # Apply thresholding
    if direction == 'keep_above':
        matrix = np.where(matrix < threshold_value, output_value(output), matrix)
    elif direction == 'keep_below':
        matrix = np.where(matrix > threshold_value, output_value(output), matrix)
    elif direction == 'keep_between' and isinstance(threshold, tuple):
        matrix = np.where((matrix < lower_threshold) | (matrix > upper_threshold), output_value(output), matrix)
    elif direction == 'exclude_between' and isinstance(threshold, tuple):
        matrix = np.where((matrix >= lower_threshold) & (matrix <= upper_threshold), output_value(output), matrix)
    
    # Apply mask mode if enabled
    if mask_mode:
        matrix = np.where((matrix != output_value(output)) & (~np.isnan(matrix)), 1, matrix)
    
    return matrix

def output_value(output_type):
    """Helper function to determine the output value based on the type."""
    return 0 if output_type == 'zero' else np.nan

def convert_index_to_coordinate(index, coordinate_affine_matrix):
    '''
    This function receives an i,j,k coordinate from a nifti and converts that to a coordinate using the destination affine. 
    
    Arguments: 
    index - i,j,k coordinate tuple
    coordinate_affine_matrix - the affine matrix of the destinate coordinate system
    voxel_mm_resolution - the spatial resolution of the voxels from the original coordinate
    
    Returns: coordinate - mni space coordinate
    '''
    mni_space_coordinate = apply_affine(coordinate_affine_matrix, index)
    return mni_space_coordinate

def convert_coordinate_to_index(mni_coords, affine):
    """
    Convert MNI coordinates to the i, j, k index tuple.
    
    Parameters:
    mni_coords (tuple): Tuple with MNI coordinates (x, y, z)
    affine (np.ndarray): Affine transformation matrix from the NIfTI header

    Returns:
    tuple: i, j, k index tuple
    """
    # Convert coordinates to voxel coordinates
    coords_np = np.append(np.array(mni_coords), 1)
    voxel_coords = np.round(np.linalg.inv(affine).dot(coords_np)[:3]).astype(int)
    return tuple(voxel_coords)

def convert_voxel_index_to_flattened_index(voxel_coords, shape):
    """
    Convert i, j, k voxel space coordinates to the index in a flattened NIfTI file.
    
    Parameters:
    voxel_coords (tuple): Tuple with i, j, k voxel space coordinates
    shape (tuple): Tuple with shape of the original NIfTI image (x, y, z)

    Returns:
    int: Index in the flattened NIfTI file
    """
    # Check if voxel coordinates are within the image shape
    if not all(0 <= coord < dim for coord, dim in zip(voxel_coords, shape)):
        raise ValueError("Voxel coordinates are out of the image bounds. Make sure you are passing voxel-space coordinates.")

    # Calculate the flattened index
    index = np.ravel_multi_index(voxel_coords, shape)

    return index

def find_maximum_voxel(matrix, mask):
    '''
    This function recevies a matrix and finds the highest valued voxel within the given brain mask
    
    Arguments: 
    matrix - matrix to be searched
    mask - brain mask to mask the search matrix
    
    Return: index of the highest valued voxel within the given brain mask
    '''
    #Use the mask to mask the matrix
    brain_mask = np.where(mask > 0, 1, 0)
    #Mask the matrix
    masked_matrix = matrix * brain_mask
    #Search for maximum voxel within the given masked brain matrix
    max_voxel = np.max(masked_matrix)
    #Find the maximum voxel's index
    indices = np.where(masked_matrix == max_voxel)

    return indices

import pandas as pd
import numpy as np

def dice_coefficient(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    '''
    Calculates the Dice Coefficient between two dataframes containing binary lesion masks.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        The first dataframe, where columns represent flattened nifti files and rows represent voxels.
        All values are zero, except for lesions which are binarized at 1.
        
    df2 : pd.DataFrame
        The second dataframe, where columns represent flattened nifti files and rows represent voxels.
        All values are zero, except for lesions which are binarized at 1.
    
    Returns:
    --------
    float
        The Dice Coefficient, a value between 0 and 1, where 1 represents a perfect overlap.
        
    '''
    # Convert the dataframes to numpy arrays
    array1 = df1.to_numpy()
    array2 = df2.to_numpy()
    
    # Calculate the intersection of non-zero elements
    intersection = np.sum(np.logical_and(array1, array2))
    
    # Calculate the number of non-zero elements in each array
    num_elements_array1 = np.sum(array1)
    num_elements_array2 = np.sum(array2)
    
    # Calculate the Dice Coefficient
    dice_coefficient = (2 * intersection) / (num_elements_array1 + num_elements_array2)
    
    return dice_coefficient

def t_to_z_array(t_values, N=None):
    """
    Convert an array of t-scores to z-scores using the inverse of the cumulative distribution function (CDF).
    
    Parameters:
    - t_values: numpy.ndarray
        Array of t-scores to be converted.
    - N: int, optional
        Number of samples from which the t-scores were derived. 
        
    Returns:
    - z_values: numpy.ndarray
        Array of converted z-scores.
        
    The function first calculates the p-values from the t-scores using the CDF of the t-distribution with
    specified degrees of freedom (df). These p-values represent the area under the curve of the t-distribution
    to the left of each t-score. Then, the p-values are converted to z-scores using the quantile function 
    (Percent-Point Function, PPF) of the standard normal distribution. The PPF is the inverse of the CDF and 
    gives the z-score that corresponds to each p-value.
    
    Note 1: This conversion is specific to statistical t-tests and should not be confused with psychometric 
    conversions of t-scores (e.g., T = 50 + 10Z), which are scaled differently.
    
    Note 2: This function does not perform the classical z-score normalization (subtract mean and divide by 
    standard deviation) across the entire DataFrame. Classical z-scoring is not applicable here because the 
    data comes from a t-distribution, not a normal distribution.
    
    This function is designed to be fast for large arrays, such as those found in neuroimaging (NIfTI files),
    where element-wise application of the conversion would be too slow.
    """
    if N is not None:
        df = N - 1  # Calculate degrees of freedom if N is provided
    else:
        raise ValueError("Either df or N must be provided.")
        
    p_values = t.cdf(t_values, df)  # Compute p-values using the CDF of the t-distribution
    z_values = norm.ppf(p_values)  # Convert p-values to z-scores using the PPF (inverse CDF) of the standard normal distribution
    
    return z_values

def convert_dataframe_t_to_z(df, sample_size=None):
    '''
    A function to convert T Score to Z scores.
    
    This works out the p-value of the corresponding T-Score by considering it's distribution. 
    To consider the T distribution, we use the sample size used to derive the T score. 
    Given the p-value derived from the T-distribution, we can calculate the corresponding Z score from the inverse CDF.
    '''
    z_scored_array = t_to_z_array(df, N=sample_size)
    df = pd.DataFrame(z_scored_array, columns=df.columns)
    return df

def fisher_z_transform(matrix, silent=True):
    if not silent:
        print('--------------------------------Performing fisher z_score-------------------------')
        print('pre fisher z score max: ', np.max(matrix), np.shape(matrix))    
    rfz_matrix = np.arctanh(matrix)
    if not silent:
        print('post fisher z score max: ', np.max(rfz_matrix), np.shape(rfz_matrix))
    return rfz_matrix

def z_score_matrix(matrix, silent=True):
    if not silent:
        print('--------------------------------Performing z_score--------------------------------')
        print('pre z score max: ', np.max(matrix), np.shape(matrix))
    z_matrix = zscore(matrix)
    if not silent:
        print('post z score max: ', np.max(z_matrix), np.shape(z_matrix))
    return z_matrix

def dice_coefficient(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    '''
    Calculates the Dice Coefficient between two dataframes containing binary lesion masks.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        The first dataframe, where columns represent flattened nifti files and rows represent voxels.
        All values are zero, except for lesions which are binarized at 1.
        
    df2 : pd.DataFrame
        The second dataframe, where columns represent flattened nifti files and rows represent voxels.
        All values are zero, except for lesions which are binarized at 1.
    
    Returns:
    --------
    float
        The Dice Coefficient, a value between 0 and 1, where 1 represents a perfect overlap.
        
    '''
    # Check if in numpy array, and convert the dataframes to numpy arrays if required
    if isinstance(df1, np.ndarray):
        array1 = df1
    else:
        array1 = df1.to_numpy()
    if isinstance(df2, np.ndarray):
        array2 = df2
    else:
        array2 = df2.to_numpy()
    
    # Calculate the intersection of non-zero elements
    intersection = np.sum(np.logical_and(array1, array2))
    
    # Calculate the number of non-zero elements in each array
    num_elements_array1 = np.sum(np.count_nonzero(array1))
    num_elements_array2 = np.sum(np.count_nonzero(array2))
    
    # Calculate the Dice Coefficient
    dice_coefficient = (2 * intersection) / (num_elements_array1 + num_elements_array2)
    
    return dice_coefficient

def mask_matrix(df_1, mask_path=None, mask_threshold=0.2, mask_by='rows'):
    '''
    input: pandas dataframe
    output: masked dataframe
    This function is expecting a dataframe to have voxels in rows, not columns
    '''
    
    #Get Mask
    if mask_path is None:
        mni_mask = nimds.get_img("mni_icbm152")
        mask_data = mni_mask.get_fdata().flatten()
        brain_indices = np.where(mask_data > 0)[0]
    else:
        mask = image.load_img(mask_path)
        mask_data = mask.get_fdata().flatten()
        brain_indices = np.where(mask_data > mask_threshold)[0]
        
    #Perform Mask
    if mask_by == 'rows':
        df_1 = df_1.iloc[brain_indices, :]
    elif mask_by == 'columns':
        df_1 = df_1.iloc[:, brain_indices]
    else:
        raise ValueError (f'unable to mask by: {mask_by}, input either rows or columns')
    
    print('Dataframes have been masked such that their shapes are: ', df_1.shape)
    return df_1

def apply_mask_to_dataframe(merged_df, mask_path=None):
    """
    Apply a mask to a dataframe using either a provided mask or the default MNI ICBM152 mask.
    
    Parameters:
    - merged_df (DataFrame): The dataframe to which the mask should be applied.
    - mask_path (str, optional): The path to the mask image. If not provided, the MNI ICBM152 mask will be used.
    
    Returns:
    - DataFrame: The masked dataframe containing only the rows specified by the mask.
    
    Example usage:
    >>> masked_df = apply_mask_to_dataframe(merged_df, mask_path=None)
    """
    
    # Load the mask data based on the provided mask_path or use default mask
    if mask_path is not None:
        brain_indices = np.where(image.load_img(mask_path).get_fdata().flatten() > 0)[0]
    else:
        mni_mask = nimds.get_img("mni_icbm152")
        mask_data = mni_mask.get_fdata().flatten()
        brain_indices = np.where(mask_data > 0)[0]
    
    # Apply the mask to the dataframe
    masked_df = merged_df.iloc[brain_indices]
    
    return masked_df

def mask_matrix(df_1, mask_path=None, mask_threshold=0.2, mask_by='rows', dataframe_to_mask_by=None):
    '''
    input: pandas dataframe
    output: masked dataframe
    This function is expecting a dataframe to have voxels in rows, not columns
    '''

    # Custom Mask
    if dataframe_to_mask_by is not None:
        mask = dataframe_to_mask_by.transpose().reset_index(drop=True).copy()
        mask['mask_index'] = mask.sum(axis=1)
        mask_indices = np.where(mask['mask_index'] != 0)[0]
        if mask_by == 'rows':
            df_1 = df_1.iloc[mask_indices, :]
        elif mask_by == 'columns':
            df_1 = df_1.iloc[:, mask_indices]
        else:
            raise ValueError(f'unable to mask by: {mask_by}, input either rows or columns')

    # MNI or Custom Mask
    else:
        # Get Mask
        if mask_path is None:
            mni_mask = nimds.get_img("mni_icbm152")
            mask_data = mni_mask.get_fdata().flatten()
            brain_indices = np.where(mask_data > 0)[0]
        else:
            mask = image.load_img(mask_path)
            mask_data = mask.get_fdata().flatten()
            brain_indices = np.where(mask_data > mask_threshold)[0]

        # Perform Mask
        if mask_by == 'rows':
            df_1 = df_1.iloc[brain_indices, :]
        elif mask_by == 'columns':
            df_1 = df_1.iloc[:, brain_indices]
        else:
            raise ValueError(f'unable to mask by: {mask_by}, input either rows or columns')

    print('Dataframes have been masked such that their shapes are: ', df_1.shape)
    return df_1


def unmask_matrix(df_1, mask_path=None, mask_threshold=0.2, unmask_by='rows', dataframe_to_unmask_by=None):
    '''
    input: pandas dataframe
    output: unmasked dataframe
    This function inserts values back into their original locations in a full brain mask.
    '''
    if isinstance(df_1, list):
        df_1 = pd.DataFrame(df_1)
    # Custom Unmask
    if dataframe_to_unmask_by is not None:
        mask = dataframe_to_unmask_by.transpose().reset_index(drop=True).copy()
        mask['mask_index'] = mask.sum(axis=1)
        mask_indices = np.where(mask['mask_index'] != 0)[0]

        # Initialize full array with NaN
        full_array = np.full(mask.shape[0], np.nan)

        # Insert values back into full array
        if unmask_by == 'rows':
            full_array[mask_indices] = df_1.values.flatten()
            unmasked_df = pd.DataFrame(full_array.reshape(1, -1))
        elif unmask_by == 'columns':
            full_array[mask_indices] = df_1.values.flatten()   # Here's the change
            unmasked_df = pd.DataFrame(full_array.reshape(-1, 1))
        else:
            raise ValueError(f'Unable to unmask by: {unmask_by}, input either rows or columns')

    # MNI or Custom Unmask
    else:
        # Get Mask
        if mask_path is None:
            mask = nimds.get_img("mni_icbm152")
            mask_data = mask.get_fdata().flatten()
            brain_indices = np.where(mask_data > 0)[0]
        else:
            mask = image.load_img(mask_path)
            mask_data = mask.get_fdata().flatten()
            brain_indices = np.where(mask_data > mask_threshold)[0]

        # Insert values back into full brain array
        if unmask_by == 'rows':
            mask_data[brain_indices] = df_1.values.flatten()
            unmasked_df = pd.DataFrame(mask_data.reshape(1, -1))
        elif unmask_by == 'columns':
            mask_data[brain_indices] = df_1.values.flatten()   # And here
            unmasked_df = pd.DataFrame(mask_data.reshape(-1, 1))
        else:
            raise ValueError(f'Unable to unmask by: {unmask_by}, input either rows or columns')

    print('Data has been unmasked to shape: ', unmasked_df.shape)
    return unmasked_df


def unmask_matrix_v2(df_1, mask_path=None, mask_threshold=0, dataframe_to_unmask_by=None):
    """
    Unmasks a matrix (given as a dataframe) back into its original full-brain form.

    Parameters:
    df_1 (DataFrame or list): The dataframe to unmask. Each NIFTI file should be in a column (by convention).
    mask_path (str, optional): Path to the mask NIFTI file. Default is MNI152 if None.
    mask_threshold (float, optional): Threshold for the mask values. Default is 0. Use 0.2 for tissue segment masks.
    dataframe_to_unmask_by (DataFrame, optional): If provided, will use this dataframe to create the mask.

    Returns:
    DataFrame: The unmasked matrix as a dataframe.

    Note:
    By convention, each NIFTI file is assumed to be a column in the dataframe.
    """

    if isinstance(df_1, pd.DataFrame) is False:
        df_1 = pd.DataFrame(df_1)
        
    # Determine the mask and its indices
    if dataframe_to_unmask_by is not None:
        mask = dataframe_to_unmask_by.copy()
        mask['mask_index'] = mask.sum(axis=1)
        mask_indices = np.where(mask['mask_index'] != 0)[0]
    else:
        if mask_path is None:
            mask = nimds.get_img("mni_icbm152")
        else:
            mask = image.load_img(mask_path)
        
        mask_data = mask.get_fdata().flatten()
        mask_indices = np.where(mask_data > mask_threshold)[0]

    # Initialize full DataFrame
    full_df = pd.DataFrame(np.nan, index=range(len(mask_data)), columns=df_1.columns)

    # Insert values back into the full DataFrame based on mask indices
    full_df.iloc[mask_indices, :] = df_1.values

    print(f'Data has been unmasked to shape: {full_df.shape}')
    return full_df

class CsvToNifti:
    """
    A class used to convert CSV files into 3D or 4D NIFTI files.

    Attributes
    ----------
    csv_files : list
        a list of CSV file paths to be converted to NIFTI.

    Methods
    -------
    load_csv_files()
        Load the CSV files into pandas DataFrames.
    convert_to_nifti()
        Convert the loaded CSV files into a 3D or 4D NIFTI file.
    save_nifti(output_file)
        Save the converted NIFTI file to the specified output file.
    """

    def __init__(self, csv_files):
        self.csv_files = csv_files
        self.data_frames = []
        self.nifti_image = None

    def unmask_csv_with_nifti_mask(csv_file, nifti_mask_file):
        # Load the CSV data and the NIFTI mask
        data_frame = pd.read_csv(csv_file)
        nifti_mask = nib.load(nifti_mask_file)

        # Convert the CSV data to a 3D array and apply the mask
        data_array = data_frame.values
        unmasked_data_array = np.where(nifti_mask.get_fdata() != 0, data_array, 0)

        # Create and return a new NIFTI image
        unmasked_nifti_image = nib.Nifti1Image(unmasked_data_array, affine=np.eye(4))
        return unmasked_nifti_image

    def load_csv_files(self):
        for csv_file in self.csv_files:
            data_frame = pd.read_csv(csv_file)
            self.data_frames.append(data_frame)

    def convert_to_nifti(self):
        data_arrays = [df.values for df in self.data_frames]
        stacked_array = np.stack(data_arrays, axis=-1)
        self.nifti_image = nib.Nifti1Image(stacked_array, affine=np.eye(4))

    def save_nifti(self, output_file):
        nib.save(self.nifti_image, output_file)


