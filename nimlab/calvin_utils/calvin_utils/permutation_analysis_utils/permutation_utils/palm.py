import statsmodels.formula.api as smf
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def permute_column(array_to_permute, looped_permutation=False):
    '''
    This function receives a numpy array and permutes it by column.
    
    Args:
    array_to_permute: A numpy array with columns to permute
    
    Returns:
    Permuted array of same shape as array_to_permute
    '''
    # Permute each patient's data
    if array_to_permute.ndim == 1:
        array_to_permute = array_to_permute.reshape(-1, 1)
    
    if looped_permutation:   
        #Permute columns
        for i in range(0, array_to_permute.shape[1]):
            array_to_permute[:,i] = np.random.permutation(array_to_permute[:,i])
        permuted_values = array_to_permute
    else:
        permuted_values = np.apply_along_axis(lambda x: np.random.permutation(x), 0, array_to_permute)
    return permuted_values

def permute_row(array_to_permute, looped_permutation=False):
    '''
    This function receives a numpy array and permutes it by row.
    
    Args:
    array_to_permute: A numpy array with rows to permute
    
    Returns:
    Permuted array of same shape as array_to_permute
    '''
    # Permute each patient's data
    if looped_permutation:
        for i in range(0, array_to_permute.shape[0]):
            array_to_permute[i,:] = np.random.permutation(array_to_permute[i,:])
            permuted_values = array_to_permute
    else:
        permuted_values = np.apply_along_axis(lambda x: np.random.permutation(x), 1, array_to_permute)
    return permuted_values

def brain_permutation(voxels_masked_to_brain, looped_permutation=False): 
    '''
    This function takes whole brains from a series of patients. 
    It will permute within each patient, then it will permute across patients
    
    Args:
    voxels_masked_to_brain: a numpy array containing the voxels from the brain
    
    Returns:
    complete_permutation: a numpy array containing the permuted voxels
    '''
    #Permute the Voxels across and within patients (PALM default)
    if looped_permutation:
        #Permute Rows
        for i in range(0, voxels_masked_to_brain.shape[0]):
            voxels_masked_to_brain[i,:] = np.random.permutation(voxels_masked_to_brain[i,:])
            
        #Permute columns
        for i in range(0, voxels_masked_to_brain.shape[1]):
            voxels_masked_to_brain[:,i] = np.random.permutation(voxels_masked_to_brain[:,i])
        complete_permutation = voxels_masked_to_brain
    else:
        permuted_within = np.apply_along_axis(lambda x: np.random.permutation(x), 1, voxels_masked_to_brain[:,:])
        complete_permutation = np.apply_along_axis(lambda x: np.random.permutation(x), 0, permuted_within[:,:]) #Improve sensitivity by doing this for each patient
    return complete_permutation

def permute_contrast_matrix(matrix_to_test, voxel_index, looped_permutation=False):
    '''
    This function permutes an entire dataframe's contrast matrix
    
    Args:
    Non-matrix_to_test: non permuted pandas dataframe with subjects in rows, 
     first several columns being patient data, and all others being voxels.
    Voxel_index: the index of the first column containing voxel data. Counted from 0. 
    looped_permutation: whether to loop over each column and row or not. Returns much better permutations, but is considerably slower.
    
    Returns:
    Permuted_dataframe
    '''
    if looped_permutation:
        #Permute non-imaging data
        permuted_non_voxel = permute_column(matrix_to_test.iloc[:,:voxel_index].to_numpy(), looped_permutation=looped_permutation)
        
        #Permute imaging data
        permuted_voxels = brain_permutation(matrix_to_test.iloc[:,voxel_index:].to_numpy(), looped_permutation=looped_permutation)
    else:
        #Permute non-voxel data
        permuted_non_voxel = np.apply_along_axis(np.random.permutation, 0, matrix_to_test.iloc[:,:voxel_index].to_numpy())
        #Permute the voxel data across and within patients (PALM default)
        permuted_voxels = brain_permutation(matrix_to_test.iloc[:,voxel_index:].to_numpy())
    
    #Reconstruct a dataframe
    permuted_dataframe = pd.DataFrame(np.concatenate((permuted_non_voxel, permuted_voxels), axis=1))
    #Set all column names to strings
    permuted_dataframe.rename(columns={col: f'vox_{col}' for col in permuted_dataframe.columns}, inplace=True)
 
    return permuted_dataframe

def whole_brain_permutation_test(matrix_to_test, t_matrix, coefficient_matrix):
    '''
    The plan will be to create one dataframe which can be used to evaluate. 
    Permuted age will go in the first column
    Connectivity will be transposed and attached to the second column
    the 'connectivity' column name will move for every voxel and the t value will be calculated
    
    args: 
    matrix_to_test is the isolated prepared dataframe to utilize.
    t_matrix is the pre-calculated matrix of t values from the non-permuted brains
    
    performs:
    Will calculate t-value for each voxel. Then, it will assess if the t-value is less than the t-value from the non-permuted brain
    If the permuted t-value is less than the t-value from the non-permuted brain, then that voxels is set to 0. 
    
    returns:
    simply a binary matrix of 1s and 0s
    '''

    # stbl code
    index_of_dependent_variable = 0
    index_of_independent_variable = 1
    index_of_first_voxel_variable = 2
    # Perform calculation of t values for each voxel
    permuted_coefficient_values = []
    permuted_t_values = []
    for i in range(len(t_matrix)):
        #The dataframe has the structure: outcome, age, voxel
        results_permuted = smf.ols(f'{matrix_to_test.columns[index_of_dependent_variable]} ~ {matrix_to_test.columns[index_of_independent_variable]}*{matrix_to_test.columns[i+index_of_first_voxel_variable]}', data=matrix_to_test).fit()
        permuted_t_values.append(results_permuted.tvalues[3])
        permuted_coefficient_values.append(results_permuted.params[3])

    #Binarize permuted t-values by comparison to non-permuted t-values
    #Any t-values that are higher are not significant, so that are = 1.
    count_of_permuted_t_higher_than_observed = np.where(np.abs(permuted_t_values)>np.abs(t_matrix), 1, 0)
    count_of_permuted_coefficient_higher_than_observed = np.where(np.abs(permuted_coefficient_values)>np.abs(coefficient_matrix), 1, 0)
    return count_of_permuted_t_higher_than_observed, count_of_permuted_coefficient_higher_than_observed