# Run a permuted dice coefficient in the cluster

import os
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import platform
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

from sklearn.decomposition import PCA

analysis = "permuted_dice_coefficient"
path_1 = r'/PHShome/cu135/python_scripts/files_to_work_on/generated_nifti.nii'
path_2 = r'/PHShome/cu135/python_scripts/files_to_work_on/0memory_network_T_map.nii'
# clin_path = 'path to clinical values'
out_dir = os.path.join(os.path.dirname(path_1), f'{analysis}')

if os.path.isdir(out_dir) != True:
    os.makedirs(out_dir)
    
from calvin_utils.file_utils.import_matrices import import_matrices_from_folder
#set file path to'' if you have specified the full path to the nifti file itself

df_1 = import_matrices_from_folder(path_1, file_pattern='')
df_2 = import_matrices_from_folder(path_2, file_pattern='')
# /Users/cu135/Dropbox (Partners HealthCare)/memory/functional_networks/ferguson_2019_networks/control_lesions/auditory_hallucination_lesions/sub-08uNodau1/roi/sub-08uNodau1_lesionMask.nii.gz

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

from calvin_utils.nifti_utils.matrix_utilities import threshold_matrix
from calvin_utils.statistical_utils.fisher_z_transform import fisher_z_transform
from nimlab import datasets as nimds

#If you want to enter a threshold (quantile) to threhsold the matrices at, enter True
threshold=True
#if you have an R map as a matrix and want to fisher transform, enter True
fish_transform=False
# if you have a whole host of matrices in df_1 or df_2, enter summate=True, otherwise summate=False (this compares 2 matrices)
summate=False

if threshold: 
    #Threshold by quantile if desirde
    quantile_target =  0.95
    
    threshold_1 = np.quantile(df_1, quantile_target)
    threshold_2 = np.quantile(df_2, quantile_target)
    
    thresholded_df_1 = threshold_matrix(df_1, threshold = threshold_1, probability=False, direction='keep_greater')
    thresholded_df_2 = threshold_matrix(df_2, threshold = threshold_2, probability=False, direction='keep_greater')
    
    thresholded_df_1[thresholded_df_1 > 0] = 1
    thresholded_df_2[thresholded_df_2 > 0] = 1
    
#Fisher transform 
if fish_transform: 
    df_1 = fisher_z_transform(df_1)

if summate:
    thresholded_df_1['for_dice'] = thresholded_df_1.sum(axis=1)
    thresholded_df_2['for_dice'] = thresholded_df_2.sum(axis=1)
else:
    thresholded_df_1['for_dice'] = thresholded_df_1
    thresholded_df_2['for_dice'] = thresholded_df_2

#Dice Coefficient Calculation
#This can only compare TWO COLUMNS. 
#Make sure you specify what column you want. 
observed_dice_coefficient = dice_coefficient(thresholded_df_1['for_dice'], thresholded_df_2['for_dice'])
print('Dice coefficient:', observed_dice_coefficient)


# Permute the Dice Coefficient
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import brain_permutation
from tqdm import tqdm 

# Assuming df_1 and df_2 are your original dataframes
n_permutations = 10000
dice_coefficients = []
voxel_index = 0
for i in tqdm(range(n_permutations)):
    # Permute dataframes
    permuted_df_1 = brain_permutation(thresholded_df_1.copy().to_numpy().reshape(1,-1), looped_permutation=True)
    permuted_df_2 = brain_permutation(thresholded_df_2.copy().to_numpy().reshape(1,-1), looped_permutation=True)

    # Threshold and calculate the Dice coefficient for the permuted dataframes
    permuted_dice_coefficient = dice_coefficient(permuted_df_1, permuted_df_2)

    # Store the Dice coefficient
    dice_coefficients.append(permuted_dice_coefficient)

# Convert the list to a numpy array
dice_coefficients = np.array(dice_coefficients)

print('empiric p: ', np.count_nonzero(dice_coefficients>observed_dice_coefficient))

empiric_p_df = pd.DataFrame({'empiric p': np.count_nonzero(dice_coefficients>observed_dice_coefficient), 'observed dice coefficient': observed_dice_coefficient}, index='value')
empiric_p_df.to_csv(os.path.join(out_dir, 'dice_coefficient_empiric_p_df.csv'))