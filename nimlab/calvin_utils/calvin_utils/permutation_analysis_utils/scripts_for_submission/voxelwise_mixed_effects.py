############################################IMPORTS########################
from calvin_utils.file_utils.print_suppression import HiddenPrints
import gc
import os
from tqdm import tqdm
import statsmodels.regression.mixed_linear_model as sm
import pandas as pd
import numpy as np
import warnings
import os
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti
from calvin_utils.nifti_utils.matrix_utilities import unmask_matrix
from patsy import dmatrices

########################################################################DEFINE INPUTS################################
out_dir = '/PHShome/cu135/permutation_tests/voxelwise_mixed_effects/pd_ad_age_stim_interaction'
path_to_data_df = '/PHShome/cu135/permutation_tests/voxelwise_mixed_effects/dataframe_for_mixed_effects.csv'
formula = 'outcome ~ Standardized_Age + voxel  + Standardized_Age:voxel'
random_effects_column = 'One_Hot_Disease'

################################################################DEFINE FUNCTIONS################################
def extract_predictors_from_formula(formula):
    y, X = dmatrices(formula, pd.DataFrame({'voxel': [0, 1], 'Standardized_Age': [0, 1], 'outcome': [0, 1]}), return_type='dataframe')
    return X.columns.tolist()[0:]

def initialize_results_array(num_voxels, expected_predictors, metrics):
    num_metrics = len(expected_predictors) * len(metrics) + 1  # +1 for 'voxel'
    return np.zeros((num_voxels, num_metrics))

def voxelwise_mixed_effects_regression_updated(data_df, formula_template, random_effects_column, model_type='linear', batch_size=50000, checkpoint_path='checkpoint.parquet', use_checkpoints=False):
    """
    Perform mixed-effects regression voxelwise based on the provided formula template.
    
    Parameters:
        data_df (pd.DataFrame): DataFrame containing outcome, voxel values, clinical covariates, and other variables with subjects in rows.
        formula_template (str): A string template for the regression formula with 'voxel' as a placeholder for voxel columns.
        voxel_columns (list): List of voxel column names in data_df.
        random_effects_column (str): The column in data_df to be used for random effects.
        model_type (str, default='linear'): Specifies the type of regression model to use ('linear' or 'logistic').
        batch_size (int, default=5000): Number of voxels to process before saving a checkpoint.
        checkpoint_path (str, default='checkpoint.parquet'): Path to save the intermediate results as a checkpoint.
        use_checkpoints (bool, default=False): whether or not to use checkpoint function

    Returns:
        results_df (pd.DataFrame): DataFrame containing p-values, coefficient values, t-values for each voxel,
                                   along with the coefficient, t-value, and p-value for each predictor.
    """
    # Extract predictors and initialize results array
    voxel_columns = data_df.columns[data_df.columns.get_loc('outcome')+1:]
    expected_predictors = extract_predictors_from_formula(formula_template)
    metrics = ['_coeff', '_t_value', '_p_value']
    num_metrics = len(expected_predictors) * len(metrics) + 1  # +1 for 'voxel'
    num_voxels = len(voxel_columns)
    results_array = np.zeros((num_voxels, num_metrics))
    
    # Existing checkpointing logic
    try:
        if (os.path.exists(checkpoint_path)) & (use_checkpoints):
            results_df = pd.read_parquet(checkpoint_path)
            start_idx = len(results_df)
        else:
            start_idx = 0
    except Exception as e:
        print(f"Failed due to error: {e}.")
    
    # Loop through each voxel column and fit the model
    for idx, voxel in enumerate(tqdm(voxel_columns[start_idx:])):
        formula = formula_template.replace('voxel', voxel)
        
        # Existing mixed-effects logic
        try:
            if model_type == 'linear':
                with HiddenPrints():
                    model = sm.MixedLM.from_formula(formula, data=data_df, groups=data_df[random_effects_column]).fit(method="cg")
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # New: Directly populate the results_array
            col_idx = 1
            for predictor in model.params.index:
                results_array[idx, col_idx:col_idx + 3] = [model.params[predictor], model.tvalues[predictor], model.pvalues[predictor]]
                col_idx += 3

        except Exception as e:
            if str(e) == "Singular matrix":
                pass  # Handle singular matrix cases as needed
            
        # Existing checkpointing logic
        try:
            if ((idx + 1) % batch_size == 0) & (use_checkpoints):
                pd.DataFrame(results_array).to_parquet(checkpoint_path)
                gc.collect()
        except Exception as e:
            print(f"Failed to save checkpoint due to error: {e}.")
    
    # Generate DataFrame from results_array
    column_names = ['voxel'] + [f"{pred}{met}" for pred in expected_predictors for met in metrics]
    results_df = pd.DataFrame(results_array, columns=column_names)
    
    return results_df

def save_results_as_nifti(results_df, out_dir, mask_path=None, mask_threshold=0.2, unmask_by='rows', dataframe_to_unmask_by=None):
    """
    Save each column in the results DataFrame as a NIFTI file.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame containing various statistical measures for each voxel.
        out_dir (str): Directory where NIFTI files should be saved.
        mask_path (str, optional): Path to the NIFTI mask file to use for unmasking.
        mask_threshold (float, optional): Mask threshold for unmasking.
        unmask_by (str, optional): Direction for unmasking ('rows' or 'columns').
        dataframe_to_unmask_by (pd.DataFrame, optional): DataFrame to use for unmasking.
        
    Returns:
        None
    """
    
    # Ensure the output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Iterate over every column in results_df and generate NIFTI files
    for colname in results_df.columns:
        # Unmask the matrix
        unmasked_df = unmask_matrix(results_df[colname], mask_path=mask_path, mask_threshold=mask_threshold,
                                    unmask_by=unmask_by, dataframe_to_unmask_by=dataframe_to_unmask_by)
        
        # Save the unmasked matrix as a NIFTI file
        view_and_save_nifti(unmasked_df, out_dir, output_name=colname)

################################################################CALLL THE FUNCTIONS################################
if __name__ == '__main__':
    data_df = pd.read_csv(path_to_data_df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_df = voxelwise_mixed_effects_regression_updated(data_df, 
                                                formula_template=formula, 
                                                random_effects_column = random_effects_column, 
                                                model_type='linear',
                                                batch_size=50000, 
                                                checkpoint_path='checkpoint.parquet'
                                                )

    results_df.to_csv(os.path.join(out_dir, 'voxelwise_mixed_effects.csv'))

    save_results_as_nifti(results_df, out_dir, mask_path=None, mask_threshold=0.2, unmask_by='rows', dataframe_to_unmask_by=None)