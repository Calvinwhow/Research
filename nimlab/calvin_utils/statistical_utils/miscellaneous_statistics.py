import numpy as np
import pandas as pd

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