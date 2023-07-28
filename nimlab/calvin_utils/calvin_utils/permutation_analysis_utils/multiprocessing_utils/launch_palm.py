#Perform analysis
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import concurrent.futures
from nimlab import datasets as nimds
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import whole_brain_permutation_test, permute_column, permute_row, permute_contrast_matrix
from calvin_utils.file_utils.dataframe_utilities import preprocess_colnames_for_regression
from calvin_utils.statistical_utils.voxelwise_statistical_testing import voxelwise_interaction_f_stat

class v1:
    #----------------------------------------------------------------Begin User Input
    # Gather information from terminal
    n_permutations = 5
    matrix_to_test = '/Users/cu135/Dropbox (Partners HealthCare)/memory/functional_networks/response_topology/voxelwise_glm/age_interaction_rios_vtas/t_values_topology/matrix_to_test.csv'
    t_matrix = '/Users/cu135/Dropbox (Partners HealthCare)/memory/functional_networks/response_topology/voxelwise_glm/age_interaction_rios_vtas/t_values_topology/t_values.csv'
    coefficient_matrix = '/Users/cu135/Dropbox (Partners HealthCare)/memory/functional_networks/response_topology/voxelwise_glm/age_interaction_rios_vtas/t_values_topology/coefficient_values.csv'
    out_dir = '/Users/cu135/Dropbox (Partners HealthCare)/memory/analyses/test_env' #input("Enter output directory: ")
    job_name = '5perms_5_workers'
    #----------------------------------------------------------------End User Input

    # Import the data
    matrix_to_test = pd.read_csv(matrix_to_test,index_col=False)
    try:
        matrix_to_test.pop('Patient # CDR, ADAS')
    except:
        print('Could not remove Patient # CDR, ADAS column')
    t_matrix = pd.read_csv(t_matrix,index_col=False)
    coefficient_matrix = pd.read_csv(coefficient_matrix,index_col=False)

    #Prepare the empiric t values matrix
    mni_mask = nimds.get_img("mni_icbm152")
    mask_data = mni_mask.get_fdata().flatten()
    brain_indices = np.where(mask_data > 0)[0]
    t_matrix = t_matrix.to_numpy()[brain_indices, -1]
    coefficient_matrix = coefficient_matrix.to_numpy()[brain_indices, -1]


    # Launch Multiprocessing of PALM Analyses
    t_p_matrix = np.zeros_like(t_matrix)
    coefficient_p_matrix = np.zeros_like(coefficient_matrix)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_permutations) as executor:
        #Begin submitting the masked data to the permutor
        results = []
        for i in tqdm(range(n_permutations), desc="Jobs Launched"):
            #This is not optimal, but it is necessary
            #Avoids identical permutations by ensuring pseudo-random permutations all occur from the same core
            permuted_matrix = permute_contrast_matrix(matrix_to_test, voxel_index=2)
    
            #Assign the permuted data to a worker. return the result
            result = executor.submit(whole_brain_permutation_test, permuted_matrix, t_matrix, coefficient_matrix)
            results.append(result)
            
            # Limit number of workers at given time to prevent memory pressure issues
            # jobs[result] = i #Add job to dict of ongoing jobs
            # if len(jobs) > max_jobs-1: #Check number of ongoing jobs
            #     completed_jobs, _ = concurrent.futures.wait(jobs.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            #     # Remove the completed job from the dict of ongoing jobs
            #     del jobs[completed_jobs.pop()]
        progress_bar = tqdm(total=n_permutations, desc="Jobs Finalized")
        for result in concurrent.futures.as_completed(results):
            #Input the permuted data into the array
            extracted_t_p_values, extracted_coefficient_p_values = result.result()
            t_p_matrix = t_p_matrix + extracted_t_p_values
            coefficient_p_matrix = coefficient_p_matrix + extracted_coefficient_p_values
            
            #Update visualization
            progress_bar.update()
        progress_bar.close()

    # Generate a filename depending on the preceding files
    i = 0
    unsaved = True
    while unsaved:
        out_file_t = os.path.join(out_dir, f"{job_name}_t_p_values_{i}.csv") # Construct the output file path
        out_file_coefficient = os.path.join(out_dir, f"{job_name}_coefficient_p_values_{i}.csv") # Construct the output file path
        if os.path.exists(out_file_t):
            i += 1
        else:
            t_p_df = pd.DataFrame(t_p_matrix) # Call the function that generates the vector
            t_p_df.to_csv(out_file_t, index=False, header=False) # Save the output to the file
    
            coefficient_p_df = pd.DataFrame(coefficient_p_matrix) # Call the function that generates the vector
            coefficient_p_df.to_csv(out_file_coefficient, index=False, header=False) # Save the output to the file
            unsaved=False
            
            
import concurrent.futures
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class PermutationTester:
    def __init__(self, function, n_permutations, matrix_to_test, function_args, observed_results, out_dir, job_name):
        self.function = function
        self.n_permutations = n_permutations
        self.matrix_to_test = matrix_to_test
        self.function_args = function_args
        self.observed_results = observed_results
        self.out_dir = out_dir
        self.job_name = job_name
        self.p_matrix = np.zeros_like(self.observed_results)

    def permute_column(self, col):
        return np.random.permutation(col)

    def permute_and_submit(self, executor):
        results = []
        for i in tqdm(range(self.n_permutations), desc="Jobs Launched"):
            # Make a copy of the input DataFrame
            dataframe_for_permutation = self.matrix_to_test.copy()
            # Permute the patient labels
            permuted_patient_labels = self.permute_column(dataframe_for_permutation.index.to_numpy())
            # Assign the permuted patient labels to the DataFrame's index
            dataframe_for_permutation.index = permuted_patient_labels
            patient_permuted_df = dataframe_for_permutation
            #Submit the matrix for calculation
            result = executor.submit(self.function, patient_permuted_df, **self.function_args)
            results.append(result)
        return results

    def run(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_permutations) as executor:
            results = self.permute_and_submit(executor)
            progress_bar = tqdm(total=self.n_permutations, desc="Jobs Finalized")
            for result in concurrent.futures.as_completed(results):
                # Input the permuted data into the array
                permuted_results = result.result()
                permuted_p_values = np.where(permuted_results > self.observed_results, 1, 0)
                self.p_matrix = self.p_matrix + permuted_p_values
                # Update visualization
                progress_bar.update()
            progress_bar.close()
        self.save_results()

    def save_results(self):
        # Generate a filename depending on the preceding files
        i = 0
        unsaved = True
        while unsaved:
            out_file = os.path.join(self.out_dir, f"{self.job_name}_p_values_{i}.csv")
            if os.path.exists(out_file):
                i += 1
            else:
                p_df = pd.DataFrame(self.p_matrix)
                p_df.to_csv(out_file, index=False, header=False)
                unsaved=False


# example usage
# tester = PermutationTester(
#     function=voxelwise_interaction_f_stat, 
#     n_permutations=5,
#     matrix_to_test_path='path_to_your_file.csv',
#     function_args={
#         'outcome_df': outcome_df,
#         'neuroimaging_dfs': neuroimaging_dfs,
#         'clinical_dfs': clinical_dfs,
#         'manual_f_stat': False,
#         'manual_g_stat': False
#     },
#     out_dir='path_to_output_directory',
#     job_name='delta_r_permutation'
# )

from sklearn.preprocessing import StandardScaler
from calvin_utils.file_utils.import_matrices import import_matrices_from_folder, import_matrices_from_csv
from calvin_utils.nifti_utils.generate_nifti import nifti_from_matrix
from nimlab import datasets as nimds
import numpy as np
from calvin_utils.statistical_utils.z_score_matrix import z_score_matrix

class VoxelwiseInteractionTester(PermutationTester):
    def __init__(self, n_permutations, outcome_path, neuroimaging_paths, clinical_paths, covariate_columns, out_dir, job_name,
                 subject_column='Patient # CDR, ADAS',
                 outcome_column='% Change from baseline (ADAS-Cog11)',
                 clinical_information_column='Age'):
        self.subject_column = subject_column
        self.outcome_column = outcome_column
        self.clinical_information_column = clinical_information_column
        self.outcome_df, self.clinical_dfs = self._prepare_dataframes(clinical_paths, subject_column, outcome_column, covariate_columns)
        self.neuroimaging_dfs = self._prepare_neuroimaging_dfs(neuroimaging_paths)

        super().__init__(
            function=voxelwise_interaction_f_stat,
            n_permutations=n_permutations,
            matrix_to_test_path=None,  # We'll handle matrix preparation within this subclass
            function_args={
                'outcome_df': self.outcome_df,
                'neuroimaging_dfs': self.neuroimaging_dfs,
                'clinical_dfs': self.clinical_dfs,
                'manual_f_stat': False,
                'manual_g_stat': False
            },
            out_dir=out_dir,
            job_name=job_name
        )

    def _prepare_dataframes(self, path, subject_column, outcome_column, covariate_columns):
        # Read the data
        data_df = pd.read_csv(path, index_col=False)

        # Prepare outcome_df
        outcomes_df = pd.DataFrame()
        outcomes_df['outcome'] = data_df.loc[:, outcome_column]
        outcomes_df['subject_id'] = data_df.loc[:, subject_column]
        outcomes_df.set_index('subject_id', inplace=True)
        outcomes_df.index = outcomes_df.index.astype(str)
        outcomes_df = preprocess_colnames_for_regression(outcomes_df)

        # Prepare clinical_dfs
        clinical_dfs = []
        for column in covariate_columns:
            clinical_df = pd.DataFrame()
            clinical_df[column] = data_df.loc[:, column]
            clinical_df['subject_id'] = data_df.loc[:, subject_column]
            clinical_df.set_index('subject_id', inplace=True)
            clinical_df.index = clinical_df.index.astype(str)
            clinical_df = preprocess_colnames_for_regression(clinical_df)
            clinical_dfs.append(clinical_df)

        return outcomes_df, clinical_dfs


    def _prepare_neuroimaging_dfs(self, paths):
        'This must be provided the csf vile containing the neuroimaging file paths of interest. Columns must be subject ID.'
        neuroimaging_dfs = []
        for path in paths:
            prepped_matrix = import_matrices_from_csv(path)

            # Set patients to those in the clinical data dataframe
            prepped_matrix = prepped_matrix.transpose()
            prepped_matrix['subject_id'] = [str(col).split('_')[0] for col in prepped_matrix.index]
            prepped_matrix.set_index('subject_id', inplace=True)
            prepped_matrix.index = prepped_matrix.index.astype(str)
            
            # Prepare column names for regression
            prepped_matrix = preprocess_colnames_for_regression(prepped_matrix)
            neuroimaging_dfs.append(prepped_matrix)
        return neuroimaging_dfs
    
#Example Usage:
# tester = VoxelwiseInteractionTester(
#     n_permutations=5,
#     outcome_path='path_to_your_outcome_file.csv',
#     neuroimaging_paths=['path_to_your_neuroimaging_file1.csv', 'path_to_your_neuroimaging_file2.csv'],
#     clinical_paths=['path_to_your_clinical_file1.csv', 'path_to_your_clinical_file2.csv'],
#     out_dir='path_to_output_directory',
#     job_name='delta_r_permutation'
# )
# tester.run()
