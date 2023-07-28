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
from calvin_utils.statistical_utils.voxelwise_statistical_testing import generate_delta_r_map

#----------------------------------------------------------------Begin User Input
# Gather information from terminal
n_permutations = 5
matrix_to_test = '/PHShome/cu135/memory/age_binarized_r_maps/csv_for_permuted_delta_r_map.csv'
empiric_matrix = '/PHShome/cu135/memory/age_binarized_r_maps/observed_delta_r_map.csv'
out_dir = '/PHShome/cu135/memory/age_binarized_r_maps/default_analysis' 
#input("Enter output directory: ")
job_name = 'delta_r_permutation'
#----------------------------------------------------------------Begin Functoin Input
threshold_of_interest = 65
column_of_interest = 'Age at DOS'

#----------------------------------------------------------------END USER INPUT----------------------------------------------------------------
# Import the data
matrix_to_test = pd.read_csv(matrix_to_test,index_col=False)
try:
    matrix_to_test.pop('Patient # CDR, ADAS')
except:
    print('Could not remove subject column')
    
#---initialize
#empiric_matrix = pd.read_csv(empiric_matrix,index_col=False)

# Launch Multiprocessing of PALM Analyses
#p_matrix = np.zeros_like(empiric_matrix)

#----------------------------------------------------------------Generate Observed Distribution
observed_delta_r = generate_delta_r_map(delta_matrix, threshold_of_interest=threshold_of_interest, column_of_interest=column_of_interest)
p_matrix = np.zeros_like(observed_delta_r)


#----------------------------------------------------------------Generated Permuted Distribution

with concurrent.futures.ProcessPoolExecutor(max_workers=n_permutations) as executor:
    #Begin submitting the masked data to the permutor
    results = []
    for i in tqdm(range(n_permutations), desc="Jobs Launched"):
        #----------------------------------------------------------------perform the permutation
        # Make a copy of the input DataFrame
        dataframe_for_permutation = matrix_to_test.copy()
        # Permute the patient labels
        permuted_patient_labels = permute_column(dataframe_for_permutation.index.to_numpy())
        # Assign the permuted patient labels to the DataFrame's index
        dataframe_for_permutation.index = permuted_patient_labels
        patient_permuted_df = dataframe_for_permutation
        
        #----------------------------------------------------------------Submit the job
        #Submit the matrix for calculation
        result = executor.submit(generate_delta_r_map, patient_permuted_df, threshold_of_interest, column_of_interest)
        results.append(result)

    progress_bar = tqdm(total=n_permutations, desc="Jobs Finalized")
    for result in concurrent.futures.as_completed(results):
        #Input the permuted data into the array
        permuted_delta_r_map = result.result()
        
        # Similarize the outputs
        if not isinstance(observed_delta_r.columns.values[0], str):
            names_list = observed_delta_r.columns.to_list()
            new_names = [str(name) for name in names_list]
            observed_delta_r.columns = new_names
        if not isinstance(permuted_delta_r_map.columns.values[0], str):
            names_list = permuted_delta_r_map.columns.to_list()
            new_names = [str(name) for name in names_list]
            permuted_delta_r_map.columns = new_names

            
        permuted_delta_r_p_values = np.where(permuted_delta_r_map > observed_delta_r, 1, 0)
        p_matrix = p_matrix + permuted_delta_r_p_values
        
        #Update visualization
        progress_bar.update()
    progress_bar.close()

# Generate a filename depending on the preceding files
i = 0
unsaved = True
while unsaved:
    out_file = os.path.join(out_dir, f"{job_name}_p_values_{i}.csv") # Construct the output file path
    if os.path.exists(out_file):
        i += 1
    else:
        p_df = pd.DataFrame(p_matrix) # Call the function that generates the vector
        p_df.to_csv(out_file, index=False, header=False) # Save the output to the file
        unsaved=False