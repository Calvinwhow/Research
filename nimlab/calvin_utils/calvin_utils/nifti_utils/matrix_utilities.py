import scipy.stats as st
import numpy as np
from nibabel.affines import apply_affine
from nimlab import datasets as nimds
from nilearn import image, plotting 

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

def threshold_matrix(matrix, threshold=0.5, probability=True, direction='bidirectional'):
    print('--------------------------------Performing threshold--------------------------------')
    if probability == True:
        high_threshold = st.norm.ppf(threshold) #this generates the z score that indicates everything BELOW the threshold value
        low_threshold = st.norm.ppf(1-threshold) #this generates th
        print('z score high threshold is: ', high_threshold)
        print('z score low threshold is: ', low_threshold)
    
    else:
        high_threshold = threshold
        low_threshold = -threshold
    if direction == 'keep_greater':
        print('pre threshold: ', np.count_nonzero(matrix))
        print('max z score in matrix is: ', np.max(matrix))
        print('min z score in matrix is: ', np.min(matrix))
        empty = np.ones(matrix.shape)
        empty[matrix < high_threshold] = 0
        matrix = matrix * empty
        print('post threshold: ', np.count_nonzero(matrix))
        print('max z score in matrix is: ', np.max(matrix))
        print('min z score in matrix is: ', np.min(matrix))
        print('I zero everything below threshold')
    elif direction == 'keep_lesser':
        matrix[matrix > low_threshold] = 0
        print('I will zero everything above threshold')
    elif direction == 'bidirectional':
        print('I will keep everything above and below the threshold')
        matrix[matrix > high_threshold] = 1; vox_over = np.sum(matrix==1)
        print('voxels over threshold: ', vox_over)
        matrix[matrix < low_threshold] = -1; vox_under = np.sum(matrix==-1)
        print('voxels under and over threshold: ', vox_under)
        matrix = np.where(abs(matrix) == 1, matrix, 0)
        print('max z score in matrix is: ', np.max(matrix))
        print('min z score in matrix is: ', np.min(matrix))
        print('voxels suerviving matrix: ', np.count_nonzero(matrix))
    else:
        print('Error encountered, direction not provided')
    return matrix

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

import numpy as np
def fisher_z_transform(matrix):
    print('--------------------------------Performing fisher z_score-------------------------')
    print('pre fisher z score max: ', np.max(matrix), np.shape(matrix))    
    rfz_matrix = np.arctanh(matrix)
    print('post fisher z score max: ', np.max(rfz_matrix), np.shape(rfz_matrix))
    return rfz_matrix

from scipy.stats import zscore
def z_score_matrix(matrix):
    print('--------------------------------Performing z_score--------------------------------')
    print('pre z score max: ', np.max(matrix), np.shape(matrix))
    z_matrix = zscore(matrix)
    z_matrix = np.nan_to_num(np.nan_to_num(z_matrix, nan=0, posinf=0, neginf=0))
    print('post z score max: ', np.max(z_matrix), np.shape(z_matrix))
    return z_matrix

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

import nibabel as nib
from nilearn import image
from nimlab import datasets as nimds
import numpy as np
import pandas as pd

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
            print(mask_data.shape)
            brain_indices = np.where(mask_data > 0)[0]
            print(brain_indices)
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

import pandas as pd
import numpy as np
import nibabel as nib

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

