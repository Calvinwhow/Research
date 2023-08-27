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
########################################################################DEFINE INPUTS################################
out_dir = '/PHShome/cu135/permutation_tests/voxelwise_mixed_effects/pd_ad_age_stim_interaction'
path_to_data_df = '/PHShome/cu135/permutation_tests/voxelwise_mixed_effects/dataframe_for_mixed_effects.csv'
formula = 'outcome ~ Standardized_Age + voxel  + Standardized_Age:voxel'



################################################################DEFINE FUNCTIONS################################
def voxelwise_mixed_effects_regression_updated(data_df, formula_template, voxel_columns, random_effects_column, model_type='linear', batch_size=50000, checkpoint_path='checkpoint.parquet', use_checkpoints=False):
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
    
    all_results = []

    # Attempt to use checkpointing
    try:
        # Check if checkpoint exists, and if so, load it and continue from the last processed voxel
        if (os.path.exists(checkpoint_path)) & (use_checkpoints):
            all_results = pd.read_parquet(checkpoint_path).to_dict('records')
            start_idx = len(all_results)
        else:
            start_idx = 0
    except Exception as e:
        print(f"Failed due to error: {e}.")

    # Loop through each voxel column and fit the model
    for idx, voxel in enumerate(tqdm(voxel_columns[start_idx:])):
        formula = formula_template.replace('voxel', voxel)
        
        try:
            if model_type == 'linear':
                with HiddenPrints():
                    model = sm.MixedLM.from_formula(formula, data=data_df, groups=data_df[random_effects_column]).fit()
            elif model_type == 'logistic':
                raise NotImplementedError("Mixed-effects logistic regression is not directly supported by statsmodels.")
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            # Extract results for each predictor and store in dictionary
            results = {'voxel': voxel}

            for predictor in model.params.index:
                results[f'{predictor}_coeff'] = model.params[predictor]
                results[f'{predictor}_t_value'] = model.tvalues[predictor]
                results[f'{predictor}_p_value'] = model.pvalues[predictor]

        except Exception as e:
            if str(e) == "Singular matrix":
                results = {'voxel': voxel}
                for predictor in formula.split("~")[1].split("+"):
                    predictor = predictor.strip()
                    results[f'{predictor}_coeff'] = np.nan
                    results[f'{predictor}_t_value'] = np.nan
                    results[f'{predictor}_p_value'] = np.nan
            else:
                raise e  # if it's another error, we raise it

        all_results.append(results)

        # Save a checkpoint if processed voxel count is a multiple of batch_size
        try:
            if ((idx + 1) % batch_size == 0) & (use_checkpoints):
                pd.DataFrame(all_results).to_parquet(checkpoint_path)
                gc.collect()
            else:
                pass
        except Exception as e:
            print(f"Failed to save checkpoint due to error: {e}. Continuing without saving checkpoint.")

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(all_results)
    
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
data_df = pd.read_csv(path_to_data_df)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results_df = voxelwise_mixed_effects_regression_updated(data_df, 
                                               formula_template=formula, 
                                               voxel_columns=data_df.columns[data_df.columns.get_loc('outcome')+1:], 
                                               random_effects_column = 'One_Hot_Disease', 
                                               model_type='linear', 
                                               batch_size=50000, 
                                               checkpoint_path='checkpoint.parquet'
                                               )

results_df.to_csv(os.path.join(out_dir, 'voxelwise_mixed_effects.csv'))

save_results_as_nifti(results_df, out_dir, mask_path=None, mask_threshold=0.2, unmask_by='rows', dataframe_to_unmask_by=None)