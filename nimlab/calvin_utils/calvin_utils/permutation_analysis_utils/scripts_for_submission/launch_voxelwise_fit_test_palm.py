"""
1) T-Test
-------------------------

This script T-statistic permutation tests on provided outcomes, clinical covariates, and voxelwise neuroimaging data. 
User input is required to define the number of permutations, the output directory, job name, and paths to the outcome, clinical covariate, and neuroimaging data. 
The script uses multiprocessing for efficient computation of the permutation tests.
It generates permuted versions of the patient labels in the input data and calculates the F-statistic for each permutation. 
The results are saved to the output directory, with each permutation result stored in a separate csv file.

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

if __name__=='__main__':

    #Imports
    import ast
    import numpy as np
    from tqdm import tqdm
    import os
    import pandas as pd
    import concurrent.futures
    from calvin_utils.permutation_analysis_utils.permutation_utils.palm import permute_column
    from calvin_utils.statistical_utils.voxelwise_statistical_testing import voxelwise_r_squared
    from calvin_utils.permutation_analysis_utils.multiprocessing_utils.memory_management import MemoryCheckingExecutor
    from calvin_utils.file_utils.script_printer import ScriptInfo
    from calvin_utils.permutation_analysis_utils.scripts_for_submission.script_descriptions import script_dict
    from concurrent.futures import ProcessPoolExecutor, as_completed


    #----------------------------------------------------------------Begin User Input
    script_info = ScriptInfo(script_dict)
    parser = script_info.create_argparse_parser('launch_voxelwise_fit_test_palm.py')
    args = parser.parse_args()
    
    # Parse the list arguments as actual lists (not strings)
    args.outcome_data_path = ast.literal_eval(args.outcome_data_path)
    args.clinical_covariate_paths = ast.literal_eval(args.clinical_covariate_paths)
    args.neuroimaging_df_paths = ast.literal_eval(args.neuroimaging_df_paths)

    print(args)
    
    #----------------------------------------------------------------END USER INPUT----------------------------------------------------------------
    #Prepare inputs and outputs
    os.makedirs(args.out_dir, exist_ok=True)

    # Prepare outcome_df
    unpermuted_outcome_df = pd.read_csv(args.outcome_data_path[0])

    # Prepare clinical_dfs containing the covariates
    clinical_dfs = []
    for path in args.clinical_covariate_paths:
        covariate_matrix = pd.read_csv(path)
        clinical_dfs.append(covariate_matrix)

    # prepare neuroimaging_dfs containing voxelwise data
    neuroimaging_dfs = []
    for path in args.neuroimaging_df_paths:
        neuroimaging_matrix = pd.read_csv(path)
        neuroimaging_dfs.append(neuroimaging_matrix)		
        
    #----------------------------------------------------------------Generate Observed Distribution
    with ProcessPoolExecutor(max_workers=int(args.n_cores)) as executor:
        # Begin submitting the masked data to the permutor
        results = []
        for i in tqdm(range(int(args.n_cores)), desc="Jobs Launched"):
            #----------------------------------------------------------------perform the permutation
            # Permute the patient labels
            permuted_patient_labels = permute_column(unpermuted_outcome_df.index.to_numpy(), looped_permutation=True).reshape(-1)
            
            outcomes_df = unpermuted_outcome_df.copy()
            outcomes_df.index = permuted_patient_labels
            for clin_df in clinical_dfs:
                clin_df.index = permute_column(clin_df.index.to_numpy(), looped_permutation=True).reshape(-1)
            for neuroimaging_df in neuroimaging_dfs:
                neuroimaging_df.index = permute_column(neuroimaging_df.index.to_numpy(), looped_permutation=True).reshape(-1)
            #----------------------------------------------------------------Submit the job
            # Submit the matrix for calculation
            result = executor.submit(voxelwise_r_squared, outcomes_df, neuroimaging_dfs, clinical_dfs)
            results.append(result)
        
        for idx, result in enumerate(concurrent.futures.as_completed(results)):
            # Input the permuted data into the array
            result_statistic = result.result()
            i = 0
            unsaved = True
            # Construct the output file path
            while unsaved:
                out_file = os.path.join(args.out_dir, f"{args.job_name}_result_values_{i}.csv")
                if os.path.exists(out_file):
                    i += 1
                else:
                    unsaved=False
                    pd.DataFrame(result_statistic).to_csv(out_file, index=False)