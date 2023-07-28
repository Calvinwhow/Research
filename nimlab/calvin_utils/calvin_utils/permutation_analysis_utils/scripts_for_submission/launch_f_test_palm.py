#Perform analysis
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import concurrent.futures
from nimlab import datasets as nimds
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import whole_brain_permutation_test
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import permute_contrast_matrix
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import permute_column
from calvin_utils.statistical_utils.voxelwise_statistical_testing import voxelwise_interaction_f_stat
from calvin_utils.file_utils.import_matrices import import_matrices_from_csv
from calvin_utils.file_utils.dataframe_utilities import preprocess_colnames_for_regression
from calvin_utils.nifti_utils.matrix_utilities import mask_matrix


#----------------------------------------------------------------Begin User Input
# Gather information from terminal
n_permutations = 5000
out_dir = '/PHShome/cu135/permutation_tests/f_test/age_by_stim_ad_dbs_redone/results/tmp'
job_name = 'ftest_bm'

clinical_data_path = '/PHShome/cu135/permutation_tests/f_test/age_by_stim_ad_dbs/grey_matter_damage_score_and_outcomes.csv'
neuroimaging_df_paths = ['/PHShome/cu135/memory/file_paths/paths_to__.csv']

outcome_column = '% Change from baseline (ADAS-Cog11)'
subject_column = 'Patient # CDR, ADAS'
covariate_columns = ['Age']
#----------------------------------------------------------------END USER INPUT----------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
# Import the clinical data
data_df = pd.read_csv(clinical_data_path)

# Prepare outcome_df
outcomes_df = pd.DataFrame()
outcomes_df['outcome'] = data_df.loc[:, outcome_column]
outcomes_df['subject_id'] = data_df.loc[:, subject_column]
outcomes_df.set_index('subject_id', inplace=True)
outcomes_df.index = outcomes_df.index.astype(str)
outcomes_df = preprocess_colnames_for_regression(outcomes_df)

# Prepare clinical_dfs containing the covariates
clinical_dfs = []
for column in covariate_columns:
	clinical_df = pd.DataFrame()
	clinical_df[column] = data_df.loc[:, column]
	clinical_df['subject_id'] = data_df.loc[:, subject_column]
	clinical_df.set_index('subject_id', inplace=True)
	clinical_df.index = clinical_df.index.astype(str)
	clinical_df = preprocess_colnames_for_regression(clinical_df)
	clinical_dfs.append(clinical_df)

neuroimaging_dfs = []
for path in neuroimaging_df_paths:
	prepped_matrix = import_matrices_from_csv(path)

	# Set patients to those in the clinical data dataframe
	prepped_matrix = prepped_matrix.transpose()
	prepped_matrix['subject_id'] = [str(col).split('_')[0] for col in prepped_matrix.index]
	prepped_matrix.set_index('subject_id', inplace=True)
	prepped_matrix.index = prepped_matrix.index.astype(str)

	# Prepare column names for regression
	prepped_matrix = preprocess_colnames_for_regression(prepped_matrix)
	neuroimaging_dfs.append(prepped_matrix)
for neuroimaging_df in neuroimaging_dfs:
    neuroimaging_df = mask_matrix(neuroimaging_df, mask_path=None, mask_threshold=0.2, mask_by='columns', dataframe_to_mask_by=None)
			
#----------------------------------------------------------------Generate Observed Distribution
with concurrent.futures.ProcessPoolExecutor(max_workers=n_permutations) as executor:
    # Begin submitting the masked data to the permutor
    results = []
    for i in tqdm(range(n_permutations), desc="Jobs Launched"):
        #----------------------------------------------------------------perform the permutation
        # Permute the patient labels
        permuted_patient_labels = permute_column(outcomes_df.index.to_numpy(), looped_permutation=True)
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