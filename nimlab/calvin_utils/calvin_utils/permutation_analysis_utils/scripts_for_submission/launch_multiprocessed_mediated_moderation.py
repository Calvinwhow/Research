import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.formula.api import ols
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from calvin_utils.statistical_utils.voxelwise_statistical_testing import voxelwise_mediated_moderation_analysis
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import permute_column

if __name__ == "__main__":
    
    # Output direcotry
    out_dir = '/PHShome/cu135/permutation_tests/mediated_moderation/results'
        
    # Get Dataframes ready
    outcomes_df = pd.read_csv('/PHShome/cu135/permutation_tests/mediated_moderation/inputs/outcomes_df.csv')
    mediator_df = pd.read_csv('/PHShome/cu135/permutation_tests/mediated_moderation/inputs/mediator_df.csv')
    moderator_df = pd.read_csv('/PHShome/cu135/permutation_tests/mediated_moderation/inputs/moderator_df.csv')
    exposure_df = pd.read_csv('/PHShome/cu135/permutation_tests/mediated_moderation/inputs/exposure_df.csv')

    permute=False

    # Define server requirements
    n_workers=1
    bootstrap_samples = 1
        
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i in tqdm(range(n_workers), desc="Jobs Launched"):
            if permute:
                permuted_patient_labels = permute_column(outcomes_df.index.to_numpy(), looped_permutation=False)
                outcomes_df.index = permuted_patient_labels
            
            result = executor.submit(voxelwise_mediated_moderation_analysis, mediator_df, moderator_df, exposure_df, outcomes_df, bootstrap_samples)
            results.append(result)

    for idx, result in enumerate(as_completed(results)):
        # Input the permuted data into the array
        result_statistic = result.result()
        i = 0
        unsaved = True
        # Construct the output file path
        while unsaved:
            out_file = os.path.join(out_dir, f"mediated_moderation_result_values_{i}.csv")
            if os.path.exists(out_file):
                i += 1
            else:
                unsaved=False
                pd.DataFrame(result_statistic).to_csv(out_file, index=False)