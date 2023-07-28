"""
1) F-Test
-------------------------

This script performs voxelwise interaction F-statistic permutation tests on provided outcomes, clinical covariates, and voxelwise neuroimaging data. 

User input is required to define the number of permutations, the output directory, job name, and paths to the outcome, clinical covariate, and neuroimaging data. 

The script uses multiprocessing for efficient computation of the permutation tests. It generates permuted versions of the patient labels in the input data and calculates the F-statistic for each permutation. The results are saved to the output directory, with each permutation result stored in a separate csv file.

Parameters:
----------
n_permutations: int
    The number of permutations to be run.

out_dir: str
    The output directory where the result csv files will be saved.

job_name: str
    The job name for identification.

outcome_data_path: str
    The path to the outcome data csv file.

clinical_covariate_paths: list of str
    The paths to the clinical covariate data csv files.

neuroimaging_df_paths: list of str
    The paths to the voxelwise neuroimaging data csv files.
"""


#Imports
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import concurrent.futures
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import permute_column
from calvin_utils.statistical_utils.voxelwise_statistical_testing import voxelwise_interaction_f_stat
from calvin_utils.permutation_analysis_utils.multiprocessing_utils.memory_management import MemoryCheckingExecutor

#----------------------------------------------------------------Begin User Input
# Gather information from terminal
n_permutations = 500
out_dir = '/PHShome/cu135/permutation_tests/f_test/age_by_stim_ad_dbs_redone/results/tmp'
job_name = 'ftest_bm'

outcome_data_path = '/PHShome/cu135/permutation_tests/f_test/age_by_stim_ad_dbs_redone/inputs/outcomes/outcome_data_1.csv'
clinical_covariate_paths = ['/PHShome/cu135/permutation_tests/f_test/age_by_stim_ad_dbs_redone/inputs/covariates/covariate_data_1.csv']
neuroimaging_df_paths = ['/PHShome/cu135/permutation_tests/f_test/age_by_stim_ad_dbs_redone/inputs/voxelwise/voxelwise_data_1.csv']

cores = 16
memory_requested = np.round(498*75)
#----------------------------------------------------------------END USER INPUT----------------------------------------------------------------
#Prepare inputs and outputs
os.makedirs(out_dir, exist_ok=True)

# Prepare outcome_df
unpermuted_outcome_df = pd.read_csv(outcome_data_path)

# Prepare clinical_dfs containing the covariates
clinical_dfs = []
for path in clinical_covariate_paths:
	covariate_matrix = pd.read_csv(path)
	clinical_dfs.append(covariate_matrix)

# prepare neuroimaging_dfs containing voxelwise data
neuroimaging_dfs = []
for path in neuroimaging_df_paths:
	neuroiamging_matrix = pd.read_csv(path)
	neuroimaging_dfs.append(neuroiamging_matrix)		
#----------------------------------------------------------------Generate Observed Distribution
with MemoryCheckingExecutor(max_workers=cores, threshold_memory_gb=memory_requested) as executor:
    # Begin submitting the masked data to the permutor
    results = []
    for i in tqdm(range(n_permutations), desc="Jobs Launched"):
        #----------------------------------------------------------------perform the permutation
        # Permute the patient labels
        permuted_patient_labels = permute_column(unpermuted_outcome_df.index.to_numpy(), looped_permutation=True)
        
        outcomes_df = unpermuted_outcome_df.copy()
        outcomes_df.index = permuted_patient_labels
        for clin_df in clinical_dfs:
            clin_df.index = permute_column(clin_df.index.to_numpy(), looped_permutation=True)
        for neuroimaging_df in neuroimaging_dfs:
            neuroimaging_df.index = permute_column(neuroimaging_df.index.to_numpy(), looped_permutation=True)
        #----------------------------------------------------------------Submit the job
        # Submit the matrix for calculation
        result = executor.submit(voxelwise_interaction_f_stat, outcomes_df, neuroimaging_dfs, clinical_dfs,
                                 model_type='linear', manual_f_stat=False, manual_g_stat=False, permutation=True)
        results.append(result)
    
    for idx, result in enumerate(concurrent.futures.as_completed(results)):
        # Input the permuted data into the array
        result_statistic = result.result()
        i = 0
        unsaved = True
        # Construct the output file path
        while unsaved:
            out_file = os.path.join(out_dir, f"f_test_result_values_{i}.csv")
            if os.path.exists(out_file):
                i += 1
            else:
                unsaved=False
                pd.DataFrame(result_statistic).to_csv(out_file, index=False)