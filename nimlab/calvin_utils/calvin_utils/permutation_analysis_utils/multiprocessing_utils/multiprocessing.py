from multiprocessing import Pool, cpu_count
import tqdm
import statsmodels.formula.api as smf
import concurrent
from tqdm import tqdm
import numpy as np
from calvin_utils.permutation_analysis_utils.perform_permutation import vector_column_permutation
from nimlab import datasets as nimds
import pandas as pd


def permute_brain(data, n_permutations):
    '''
    This function takes a dataframe and a number of permutations.
    It will return a 3-dimensional numpy array with the index permuted for each observation. 
    
    This suffers from RAM overloading, which will crash the kernel. It can be optimized by passing integeres instead of floats.
    '''
    
    # Load the brain mask
    mni_mask = nimds.get_img("mni_icbm152")
    mask_data = mni_mask.get_fdata().flatten()

    # Get the indices of the brain voxels
    brain_indices = np.where(mask_data > 0)[0]
    
    # Get the voxels to be permuted
    indices_to_permute = data[brain_indices, :]
    
    # Create a 3D numpy array to hold the permuted data
    permuted_indices= np.zeros((indices_to_permute.shape[0], indices_to_permute.shape[1], n_permutations))
    
    #----THIS IS GOOD CODE TO MULTIPROCESS AND MITIGATE MEMORY PRESSURE--------------------------------
    # Resource informaiton
    max_jobs = 50
    jobs = {}
    # Permute the data independently of eachother. May need to copy data to prevent read errors.
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        #Begin submitting the masked data to the permutor
        results = []
        for i in tqdm(range(n_permutations), desc="Jobs Launched"):
            #Assign the permuted data to a worker. return the result
            # result = executor.submit(permute_col, indices_to_permute)
            result = executor.submit(vector_column_permutation, indices_to_permute)
            results.append(result)
            
            # Limit number of workers at given time to prevent memory pressure issues
            # jobs[result] = i #Add job to dict of ongoing jobs
            # if len(jobs) > max_jobs-1: #Check number of ongoing jobs
            #     completed_jobs, _ = concurrent.futures.wait(jobs.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            #     # Remove the completed job from the dict of ongoing jobs
            #     del jobs[completed_jobs.pop()]

        progress_bar = tqdm(total=permuted_indices.shape[2], desc="Permutations Placed in Array")
        for i, result in enumerate(concurrent.futures.as_completed(results)):
            #Input the permuted data into the array
            permuted_indices[:, :, i] = result.result()
            #Update visualization
            progress_bar.update()
        progress_bar.close()
    return permuted_indices


def process_voxel(i, prepped_matrix, permuted_array, data_df):
    #Do not calculate on zero values as intercept will be applied in the regressoin
    if np.sum(prepped_matrix.iloc[i,:]) != 0:
        #Assign a temporary dataframe with values that the statsmodels model is expecting
        ##the plan is to use patient ID to intersect the voxelwise data to the associated clinical data
        temp_df = data_df.copy()
        #Names of the columns are as per data_df. The voxel coumn's name is the integer of the voxel
        temp_df = temp_df.merge(prepped_matrix.iloc[i,:].transpose(), left_index=True, right_index=True)
        #Rename the column of the voxel to 'connectivity' to prepare it for the statsmodels input requirements
        temp_df = temp_df.rename(columns={i: "connectivity"}, errors="raise")
        
        #Fit model. Outcomes are always first
        results = smf.ols(f'{temp_df.columns[0]} ~ {temp_df.columns[1]}*{temp_df.columns[2]}', data=temp_df).fit()
        #Extract the t statistic of relevance from the model at this voxel. Interaction effect is len(indep_vars)+1 (intercept, main1, main2, interaction)
        voxel_t_value = results.tvalues[3]
        
        #Assess the t value against the permutation's distribution 
        t_value_list = []       
        for j in range(permuted_array.shape[2]):
            # Assign a temporary dataframe with values that the statsmodels model is expecting
            temp_df_permuted = data_df.copy()
            #Permute the associated data
            temp_df_permuted[temp_df.columns[1]] = np.random.permutation(temp_df[temp_df.columns[1]])
            #Retrive the already permuted voxel for all patients at this permutation
            temp_df_permuted[temp_df.columns[2]] = permuted_array[i,:,j]
            
            # Fit model. Outcomes are always first
            results_permuted = smf.ols(f'{temp_df_permuted.columns[0]} ~ {temp_df_permuted.columns[1]}*{temp_df_permuted.columns[2]}', data=temp_df_permuted).fit()
            # Extract the t statistic of relevance from the model at this voxel. Interaction effect is len(indep_vars)+1 (intercept, main1, main2, interaction)
            voxel_t_value_permuted = results_permuted.tvalues[3]
            t_value_list.append(voxel_t_value_permuted)
            
        #Calculate the p-value from the observed t-value copmared to the empiric permuted distribution of t-values
        voxelwise_p_value = calculate_empirical_p(voxel_t_value_permuted, t_value_list)
    else:
        #If voxel is zero-connectivity, assign zero so as to avoid application of intercept
        voxelwise_p_value = 0
    return voxelwise_p_value
        
def calculate_empirical_p(observed_t_value, sim_t_values):
    """
    Calculate empirical p-value from a distribution of empiric t-values and an observed t-value.
    """
    # count the number of simulated t-values that are as extreme or more extreme than the critical t-value
    extreme_count = np.sum(np.abs(sim_t_values) >= observed_t_value)

    # divide the number of extreme values by the total number of simulated values to get the empirical p-value
    empirical_p = extreme_count / len(sim_t_values)

    return empirical_p

def initialize(permuted_array, prepped_matrix, data_df):
    p_values_list = []
    if __name__ == '__main__':
        with Pool(processes=cpu_count()) as pool:
            p_values_list = list(tqdm.tqdm(pool.imap(process_voxel, range(0, len(prepped_matrix)), args={permuted_array, prepped_matrix, data_df}), total=len(prepped_matrix)))
    else:
        with Pool(processes=cpu_count()) as pool:
            p_values_list = list(tqdm.tqdm(pool.imap(process_voxel, range(0, len(prepped_matrix)), args={permuted_array, prepped_matrix, data_df}), total=len(prepped_matrix)))
    return p_values_list


def permute_col(indices_to_permute):
    '''
    This function receives a list of indices and returns them after permuting the data
    '''
    # Permute each patient's data
    permuted_values = np.empty(indices_to_permute.shape)
    for i in range(indices_to_permute.shape[1]):
        permuted_values[:,i] = indices_to_permute[:,i][np.random.permutation(len(indices_to_permute))]
    return permuted_values

def calculate_t_value(df):
    # The dataframe has the structure: outcome, age, voxels
    results_permuted = smf.ols(f'{df.columns[0]} ~ {df.columns[1]}*{df.columns[2]}', data=df).fit()
    return results_permuted.tvalues[2]

def whole_brain_permutation_test(matrix_to_test, t_matrix):
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
    #Permute outcomes
    permuted_outcomes = np.apply_along_axis(np.random.permutation, 0, matrix_to_test.iloc[:,0].values)
    #Permute the age
    permuted_age = np.apply_along_axis(np.random.permutation, 0, matrix_to_test.iloc[:,1].values)    
    #Permute the Voxels across and within patients (PALM default)
    permuted_within = np.apply_along_axis(lambda x: np.random.permutation(x), 1, matrix_to_test.iloc[:,2:].values)
    permuted_across = np.apply_along_axis(np.random.permutation, 0, permuted_within) #Improve sensitivity by doing this for each patient
    #Reconstruct a dataframe
    permuted_dataframe = pd.DataFrame(np.concatenate((permuted_outcomes.reshape(-1, 1), permuted_age.reshape(-1, 1), permuted_across), axis=1))
    #Set all column names to strings
    permuted_dataframe.rename(columns={col: f'vox_{col}' for col in permuted_dataframe.columns}, inplace=True)
    
    # stbl code
    # Perform calculation of t values for each voxel
    permuted_t_values = []
    for i in range(len(t_matrix)):
        #The dataframe has the structure: outcome, age, voxels
        results_permuted = smf.ols(f'{permuted_dataframe.columns[0]} ~ {permuted_dataframe.columns[1]}*{permuted_dataframe.columns[i+2]}', data=permuted_dataframe).fit()
        permuted_t_values.append(results_permuted.tvalues[3])

    #Binarize permuted t-values by comparison to non-permuted t-values
    #Any t-values that are higher are not significant, so that are = 1.
    return np.where(permuted_t_values>t_matrix, 1, 0)