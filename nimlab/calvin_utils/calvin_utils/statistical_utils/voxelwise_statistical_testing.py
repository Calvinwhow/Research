import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.stats.api import anova_lm
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import chi2, t, f
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import pandas as pd 
import nibabel as nib
from nimlab import datasets as nimds

from calvin_utils.nifti_utils.matrix_utilities import unmask_matrix, mask_matrix
from sklearn.linear_model import LinearRegression


def generate_interaction_features(df):
    """Generate interaction features between every column in a dataframe and appends them to a new column"""
    new_df = pd.DataFrame()
    columns = df.columns
    
    for c1, c2 in itertools.combinations(columns, 2):
        feature_name = f"{c1}_x_{c2}"
        new_df[feature_name] = df[c1] * df[c2]
        
    return pd.concat([df, new_df], axis=1)


def calculate_confidence_intervals(ab_paths, mediators):
    """
    Calculates the confidence intervals and p-value based on the bootstrapped samples.

    Parameters:
    - ab_paths: list of lists containing the bootstrapped ab paths for each mediator.
    - total_indirect_effects: list of bootstrapped summed ab paths.
    - mediators: list of mediator names.

    Returns:
    - DataFrame with the mean indirect effect, confidence intervals, and p-values for each mediator and the total indirect effect.
    """
    ab_path_values = np.array(ab_paths)

    # Check if there's only one mediator
    if isinstance(mediators, str):
        mediators = [mediators]

    # Calculate mean indirect effect and confidence intervals for each mediator
    mean_ab_paths = np.mean(ab_path_values, axis=0)
    lower_bounds = np.percentile(ab_path_values, 2.5, axis=0)
    upper_bounds = np.percentile(ab_path_values, 97.5, axis=0)

    # Calculate p-values for each mediator
    if len(mediators) == 1:
        ab_path_p_values = np.mean(np.sign(mean_ab_paths) * ab_path_values <= 0)
    else:
        ab_path_p_values = [np.mean(np.sign(mean_ab_paths[i]) * ab_path_values[:, i] <= 0) for i in range(len(mean_ab_paths))]

    # Create DataFrame to store the results
    result_df = pd.DataFrame({
        'Point Estimate': mean_ab_paths,
        '2.5th Percentile': lower_bounds,
        '97.5th Percentile': upper_bounds,
        'P-value': ab_path_p_values
    }, index=mediators)

    return result_df

def perform_mediated_moderation_analysis(dataframe, exposure, mediator, moderator, dependent_variable, bootstrap_samples=5000):
    """
    Performs a mediated moderation analysis by estimating the joint indirect effects of an exposure variable
    through a mediator on a dependent variable using bootstrapping, considering the moderating effect of another variable.

    Parameters:
    - dataframe: DataFrame containing the data.
    - exposure: str, column name of the exposure variable.
    - mediator: str, column name of the mediator variable.
    - moderator: str, column name of the moderator variable.
    - dependent_variable: str, column name of the dependent variable.
    - bootstrap_samples: int, optional, number of bootstrap samples to be used (default is 5000).

    Returns:
    - DataFrame with the mean indirect effect, confidence intervals, and p-values for the indirect effect.

    Example Usage:
    result_df = perform_mediated_moderation_analysis(data_df, exposure='Age',
                                                     mediator='Brain_Lobe',
                                                     moderator='Stimulation',
                                                     dependent_variable='Outcome')
    """

    ab_paths = []

    # Loop over each bootstrap sample
    for i in range(bootstrap_samples):
        # Resample the data with replacement
        sample = dataframe.sample(frac=1, replace=True)

        # Fit the models and calculate the indirect effect for this bootstrap sample
        model_M = smf.ols(f"{mediator} ~ {moderator}", data=sample).fit()
        model_Y = smf.ols(f"{dependent_variable} ~ {exposure} + {mediator} + {moderator} + {exposure}:{mediator} + {exposure}:{moderator}", data=sample).fit()

        indirect_effect = model_M.params[moderator] * model_Y.params[f'{exposure}:{mediator}']

        # Append the indirect effect to the list
        ab_paths.append(indirect_effect)
    # Calculate confidence intervals and p-values
    if bootstrap_samples==1:
        return pd.DataFrame(ab_paths, columns=['Indirect Effect'])
    else:
        return calculate_confidence_intervals(ab_paths, mediators=mediator)

def voxelwise_mediated_moderation_analysis(mediator_df, moderator_df, exposure_df, outcome_df, bootstrap_samples=5000):
    """
    Perform voxelwise mediated moderation analysis between the corresponding voxels from
    neuroimaging dataframes and clinical dataframes on a patient's outcome.
    
    Parameters:
        mediator_df (pd.DataFrame): DataFrame containing the mediator variable with patients in rows and 'subject_id' as index.
        moderator_df (pd.DataFrame): DataFrame containing the moderator variable with patients in rows and 'subject_id' as index.
        exposure_df (pd.DataFrame): DataFrame containing the exposure variable with patients in rows and 'subject_id' as index.
        outcome_df (pd.DataFrame): DataFrame containing the outcome variable in 'outcome' column with patients in rows and 'subject_id' as index.
        bootstrap_samples (int): Number of bootstrap samples to use in mediated moderation analysis.
    
    Returns:
        results_df (pd.DataFrame): DataFrame containing mediated moderation analysis results for each voxel.
    """
    
    # Number of voxels
    n_voxels = exposure_df.shape[1]

    # Initialize a list to store the results for each voxel
    results = []

    # Loop through each voxel and perform mediated moderation analysis
    for i in tqdm(range(n_voxels)):
        # Create temporary dataframe with outcome, mediator, moderator, and exposure data for the current voxel
        temp_df = pd.DataFrame({
            'exposure': exposure_df.iloc[:, i] if exposure_df.shape[1] > 1 else exposure_df.iloc[:, 0],
            'mediator': mediator_df.iloc[:, i] if mediator_df.shape[1] > 1 else mediator_df.iloc[:, 0],
            'moderator': moderator_df.iloc[:, i] if moderator_df.shape[1] > 1 else moderator_df.iloc[:, 0],
            'outcome': outcome_df['outcome']
        })

        # Perform mediated moderation analysis on the temporary dataframe
        result_df = perform_mediated_moderation_analysis(dataframe = temp_df,
                                                         exposure = 'exposure', 
                                                         mediator = 'mediator', 
                                                         moderator = 'moderator', 
                                                         dependent_variable ='outcome', 
                                                         bootstrap_samples=bootstrap_samples)
        # Append voxel index to result_df
        result_df['voxel_index'] = i
        results.append(result_df)
    results_df = pd.concat(results)
    return results_df

def calculate_g_statistic(full_model, reduced_model):
    #### WORK IN PROGRESS #####
    """
    Calculates the G-statistic for a given model.
    Please note, a G-statistic is comparable to various other statistics under various conditions, 
    but it is not meant to derive p-values analytically. It is meant to derive p-values using permutation testing. 

    Parameters:
        full_model: A fitted model object from the full model.
        reduced_model: A fitted model object from the reduced model.
        heteroscedastic: Boolean. If True, the errors are assumed to be heteroscedastic, 
                         and a chi-square distribution is used to calculate the p-value. 
                         If False, the errors are assumed to be homoscedastic and a Student's 
                         t-distribution is used to calculate the p-value.

    Returns:
        G_statistic: The calculated G-statistic.
        p_value: The p-value associated with the G-statistic.
    """
    # ψ^: The estimated parameters from the full model, minus those from the reduced model
    psi_hat = full_model.params - reduced_model.params
    # C: The contrast matrix. This depends on your specific hypotheses and model.
    # Assuming that full_model and reduced_model are statsmodels regression result objects
    # Get the parameter names from both models
    full_model_params = full_model.params.index
    reduced_model_params = reduced_model.params.index

    # Initialize the contrast matrix as a zero matrix with the length of the full model params
    C = np.zeros(len(full_model_params))

    # For each parameter in the full model, if it is not in the reduced model, set the corresponding
    # element in the contrast matrix to 1
    for i, param in enumerate(full_model_params):
        if param not in reduced_model_params:
            C[i] = 1

    # Ensure that C remains a 2D array (i.e., a matrix), which is expected for matrix operations
    C = C.reshape(-1, len(C))

    # M: The design matrix from the full model
    M = full_model.model.exog

    # W: Diagonal weighting matrix.
    # Compute the residuals from your full model
    residuals = full_model.resid.to_numpy()

    # Compute the variance of residuals
    variances = np.reshape(np.var(residuals), -1)

    # Assume gn contains the variance group assignments for each observation
    # Assume R is the residual forming matrix
    # Assume epsilon_hat contains the vector of residuals

    # Initialize W as a zero matrix with the same shape as R
    W = np.zeros_like(residuals)

    # Iterate over each observation
    for n in range(len(gn)):
        # Get the variance group assignment for the n-th observation
        variance_group = gn[n]

        # Find the indices of observations belonging to the same variance group
        group_indices = np.where(gn == variance_group)[0]

        # Compute the sum of diagonal elements of R for the variance group
        sum_R = np.sum(R[group_indices, group_indices])

        # Compute the product of epsilon_hat for the variance group
        product_epsilon_hat = np.prod(epsilon_hat[group_indices])

        # Compute the diagonal element of W for the n-th observation
        W_nn = sum_R / product_epsilon_hat

        # Set the diagonal element of W for the n-th observation
        W[n, n] = W_nn


    # Calculate Λ and the inverse of it.
    Lambda_inv = np.linalg.inv(C @ M.T @ W @ M @ C.T)

    # Calculate the G-statistic
    G_statistic = psi_hat.T @ C.T @ Lambda_inv @ C @ psi_hat

    # Degrees of freedom is the rank of C
    df = np.linalg.matrix_rank(C)

    # Compute 1-tailed p-value to assess if full model is significant better than reduced model
    # Assess heteroscedasticity with the Breusch-Pagan test, p-value <0.05 indicates the linear model is heteroscedastic
    _, p_value, _, _ = het_breuschpagan(residuals, full_model.model.exog)
    print('6')
    if p_value < 0.05:
        if df == 1:
            p_value = np.NaN
            #This is equivalent to Welch's v^2, which does not have an analytical distribution
        else:
            p_value = np.NaN
            #This is equivalent to Aspen-Welch v, which does not have an analytical distribution
        print('7')
    else:
        print('8')
        if df == 1:
            print('9')
            p_value = 2 * (1 - t.cdf(np.sqrt(G_statistic), df))
            #This is equivalent to student's T, which does have an analytical distribution
        else:
            print('10')
            p_value = 1 - f.cdf(G_statistic, df, full_model.df_resid - df)
            #This is equivalent to F-ratio, which does have an analytical distribution

    return G_statistic, p_value



def handle_nan_p_values(p_value_series):
    """
    Function to handle NANs in p-values by backward filling.
    
    Parameters:
        p_value_series (pd.Series): Series containing the p-values.
    
    Returns:
        p_value_series (pd.Series): Series with NANs handled.
    """
    
    if p_value_series.isna().any():
        p_value_series.fillna(method='bfill', inplace=True)
        print('WARNING: p-values containing NAN values')
        if p_value_series.isna().any():
            p_value_series.fillna(1.0, inplace=True)
    
    return p_value_series


def fdr_correct_p_values_and_threshold_r_squared(results_df, alpha=0.05):
    """
    Performs FDR correction on the p-values in the results DataFrame and thresholds R-squared values based on the corrected p-values.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame containing the p-values and R-squared values.
        alpha (float): Significance level for FDR correction.
        
    Returns:
        results_df (pd.DataFrame): DataFrame with FDR corrected p-values and thresholded R-squared values.
    """
    
    # Correct for multiple comparisons using FDR
    p_values = results_df['one_way_p_value'].values
    _, pvals_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # Add FDR corrected p-values to the DataFrame
    results_df['fdr_corrected_p_value'] = pvals_fdr
    
    # Threshold R-squared values based on FDR corrected p-values
    try:
        results_df['adj_r_squared_thresholded'] = results_df['adj_r_squared']
        results_df.loc[results_df['fdr_corrected_p_value'] > alpha, 'adj_r_squared_thresholded'] = 0.0
    except:
        print(' results_df["adj_r_squared_thresholded"]  not found')
    
    return results_df

def voxelwise_interaction_f_stat(outcome_df, predictor_neuroimaging_dfs, predictor_clinical_dfs, model_type='linear', manual_f_stat=False, manual_g_stat=True, permutation=False):
    """
    Perform voxelwise regression with interactions between the corresponding voxels from
    neuroimaging dataframes and clinical dataframes on a patient's outcome and use F-test to compare models with and without interactions.
    The F-test is the proportion of mean squared errors. When comparing two different models, F-statistic is the proprtion of the change in
    the mean squared error between the two models compared to the full model. 
    However, this means that if the second model is much worse than the first model (Larger MSE), the F-statistic can be negative. Thus, 
    negative F-statistic is considered to be equivalent to and F-statistic of zero. 
    
    degrees_of_freedom is equivalent to sample size. 
    F-statistic = ((sum_squared_residuals_1 - sum_squared_residuals_2)/(df_residuals_1 - df_residuals_2)) / (sum_squared_residuals_2/(df_residuals_2)
    MSE = sum_squared_residuals/degrees_of_freedom 
    F-statistic = delta_MSE/full_model_MSE
    
    Parameters:
        outcome_df (pd.DataFrame): DataFrame containing the outcome variable in 'outcome' column with patients in rows and 'subject_id' as index.
        predictor_neuroimaging_dfs (list of pd.DataFrame): List of DataFrames containing voxelwise neuroimaging data with patients in rows and 'subject_id' as index.
        predictor_clinical_dfs (list of pd.DataFrame): List of DataFrames containing clinical data with patients in rows and 'subject_id' as index.
        model_type (str): Specifies the type of regression model to use ('linear' or 'logistic').
        manual_f_stat (bool): If True, use the manual calculation for F-statistic and p-value. Otherwise, use anova_lm function.
    
    Returns:
        results_df (pd.DataFrame): DataFrame containing F-statistics and p-values for each voxel.
        
    Cite: 
    chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://sites.duke.edu/bossbackup/files/2013/02/FTestTutorial.pdf
    """
    
    # Number of voxels in the first neuroimaging dataframe
    n_voxels = predictor_neuroimaging_dfs[0].shape[1]

    # Initialize a list to store the results for each voxel
    results = []

    # Define lambda functions for standardization
    standardize_within_patient = lambda df: df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    standardize_across_patients = lambda df: df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # Standardize outcome data across patients
    outcome_df = standardize_across_patients(outcome_df)
    if predictor_neuroimaging_dfs[0] is not None:
        # Standardize neuroimaging data within each patient
        predictor_neuroimaging_dfs = [standardize_within_patient(df) for df in predictor_neuroimaging_dfs]
    if predictor_clinical_dfs[0] is not None:
        # Standardize clinical data across patients
        predictor_clinical_dfs = [standardize_across_patients(df) for df in predictor_clinical_dfs]

    # Loop through each voxel and perform regression
    for i in tqdm(range(0, n_voxels)):
        # Create temporary dataframe with outcome and corresponding voxel from all neuroimaging dataframes and clinical data
        temp_df = outcome_df[['outcome']].copy()
        variable_names = []
        
        if predictor_neuroimaging_dfs[0] is not None:
            for j, neuroimaging_df in enumerate(predictor_neuroimaging_dfs):
                temp_df[f'dataframe_{j}_voxel_i'] = neuroimaging_df.iloc[:, i]
                variable_names.append(f'dataframe_{j}_voxel_i')
        
        if predictor_clinical_dfs[0] is not None:
            for j, clinical_df in enumerate(predictor_clinical_dfs):
                temp_df = temp_df.merge(clinical_df, left_index=True, right_index=True)
                variable_names.extend(clinical_df.columns.tolist())
            
        # Construct the regression formulas dynamically
        variables_combined = " + ".join(variable_names)
        interaction_terms = " + ".join([f"{var1}:{var2}" for idx, var1 in enumerate(variable_names) for var2 in variable_names[idx+1:]])
        
        formula_no_interaction = f'outcome ~ {variables_combined}'
        formula_interaction = f'outcome ~ {variables_combined} + {interaction_terms}'
                
        # Fit the models
        if model_type == 'linear':
            model_no_interaction = smf.ols(formula_no_interaction, data=temp_df).fit()
            model_interaction = smf.ols(formula_interaction, data=temp_df).fit()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if manual_f_stat:
            # Calculate the F-statistic manually
            sum_square_1 = model_no_interaction.ssr
            sum_square_2 = model_interaction.ssr
            degrees_freedom_1 = model_no_interaction.df_resid
            degrees_freedom_2 = model_interaction.df_resid

            F_statistic = ((sum_square_1 - sum_square_2) / (degrees_freedom_1 - degrees_freedom_2)) / (sum_square_2 / degrees_freedom_2)
            if F_statistic < 0:
                F_statistic = 0 
            # Calculate the p-value using the cumulative distribution function of the F-distribution
            if np.isinf(F_statistic):
                print("Warning: Infinite F-statistic detected. Setting p-value to zero.")
                P_value = 0
            else:
                P_value = 1 - f.cdf(F_statistic, degrees_freedom_1 - degrees_freedom_2, degrees_freedom_2)
            statistic = 'manual_f_statistic'
        elif manual_g_stat:
            # This assesses heteroscedasticity at every voxel and decides what distribution the G-statistic should use
            f_stat = calculate_g_statistic(model_interaction, model_no_interaction)
            statistic = 'manual_g_statistic'
        else:
            # Calculate the F-statistic and p-value using anova_lm function
            table = anova_lm(model_no_interaction, model_interaction)
            F_statistic = table['F'][1]
            if np.isinf(F_statistic):
                print("Warning: Infinite F-statistic detected. Setting p-value to zero.")
                P_value = 0.0
            else:
                P_value = table['Pr(>F)'][1]
            statistic = 'statsmodels_f_statistic'
        # Store the results for the current voxel
        voxel_results = {
            'voxel_index': i,
            'statistic': F_statistic,
            'statistic_method': statistic,
            'one_way_p_value': P_value,
            'unc_r_squared': model_interaction.rsquared,
            'adj_r_squared': model_interaction.rsquared_adj
        }
        # Append the voxel_results dictionary to the results list
        results.append(voxel_results)

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    print('\n Example formula without interaction: ', formula_no_interaction)
    print('Example formula with interaction: ', formula_interaction)
    if permutation:
        return results_df['statistic']
    else:
        return results_df['statistic'], results_df, temp_df
    
def voxelwise_r_squared(outcome_df, predictor_neuroimaging_dfs, predictor_covariate_dfs, get_coefficients=False):
    """
    Perform voxelwise regression using numpy arrays and Statsmodels, calculating R-squared for each voxel.
    
    Parameters:
        outcome_df (pd.DataFrame): DataFrame containing the outcome variable with observations in rows.
        predictor_neuroimaging_dfs (list of pd.DataFrame): List of DataFrames with neuroimaging data, observations in rows, voxels in columns.
        predictor_covariate_dfs (list of pd.DataFrame): List of DataFrames with covariate data, observations in rows.
        get_coefficients (bool): Boolean regarding whether or not to calculate/return coefficients. 

    Returns:
        results_df (pd.DataFrame): DataFrame containing R-squared for each voxel.
    """
    
    # Extract outcome data from DataFrame and standardize it
    y = outcome_df.iloc[:, 0].values[:, np.newaxis]  # dynamically access the first (and only) column
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std
    
    # Concatenate all covariate arrays and standardize
    if predictor_covariate_dfs:
        covariate_arrays = [df.values for df in predictor_covariate_dfs]
        X_covariates = np.hstack(covariate_arrays)
    else:
        X_covariates = np.empty((len(y), 0))  # No covariates
        
    # Initialize a list to store the results for each voxel
    results = {}
    if get_coefficients:
        coefficients = {}
    
    # Loop through each voxel in the neuroimaging data
    for voxel in tqdm((predictor_neuroimaging_dfs[0].index), desc="Calculating R-squared for each voxel"):
        # Gather data for this voxel from all neuroimaging dataframes (assuming each has the same structure)
        X_voxel = np.hstack([df.loc[voxel, :].values[:, np.newaxis] for df in predictor_neuroimaging_dfs])
        # Combine covariates and voxel-specific data
        X = np.hstack((X_covariates, X_voxel))

        # Fit the linear regression model
        model = OLS(y, X).fit()
        
        # Collect R-squared
        results[voxel] = {'R_squared': model.rsquared}
        if get_coefficients:
            results[voxel] = {'R_squared': model.rsquared, 'Coefficients': model.params}

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    return results_df

    
#--------DEVELOPMENT--------
def voxelwise_interaction_t_stat(outcome_df, predictor_neuroimaging_dfs, predictor_clinical_dfs, model_type='linear', manual_t_stat=False, permutation=False, comprehensive_results=False):
    """
    Perform voxelwise regression with interactions between the corresponding voxels from
    neuroimaging dataframes and clinical dataframes on a patient's outcome.
    and use F-test to compare models with and without interactions and extract T-statistic for coefficients. 
    
    The T-statistic of a given coefficient is the Mean Squared Residual divided by the Mean Squared Error of that 
    
    Parameters:
        outcome_df (pd.DataFrame): DataFrame containing the outcome variable in 'outcome' column with patients in rows and 'subject_id' as index.
        predictor_neuroimaging_dfs (list of pd.DataFrame): List of DataFrames containing voxelwise neuroimaging data with patients in rows and 'subject_id' as index.
        predictor_clinical_dfs (list of pd.DataFrame): List of DataFrames containing clinical data with patients in rows and 'subject_id' as index.
        model_type (str): Specifies the type of regression model to use ('linear' or 'logistic').
    
    Returns:
        results_df (pd.DataFrame): DataFrame containing t-statistics and p-values for each voxel.
        
    Cite: 
    chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://sites.duke.edu/bossbackup/files/2013/02/FTestTutorial.pdf
    """
    
    # Number of voxels in the first neuroimaging dataframe
    n_voxels = predictor_neuroimaging_dfs[0].shape[1]

    # Initialize a list to store the results for each voxel
    results = []

    # Define lambda functions for standardization
    standardize_within_patient = lambda df: df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    standardize_across_patients = lambda df: df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # Standardize outcome data across patients
    outcome_df = standardize_across_patients(outcome_df)
    if predictor_neuroimaging_dfs[0] is not None:
        # Standardize neuroimaging data within each patient
        predictor_neuroimaging_dfs = [standardize_within_patient(df) for df in predictor_neuroimaging_dfs]
    if predictor_clinical_dfs[0] is not None:
        # Standardize clinical data across patients
        predictor_clinical_dfs = [standardize_across_patients(df) for df in predictor_clinical_dfs]

    # Loop through each voxel and perform regression
    for i in tqdm(range(0, n_voxels)):
        # Create temporary dataframe with outcome and corresponding voxel from all neuroimaging dataframes and clinical data
        temp_df = outcome_df[['outcome']].copy()
        variable_names = []
        
        if predictor_neuroimaging_dfs[0] is not None:
            for j, neuroimaging_df in enumerate(predictor_neuroimaging_dfs):
                temp_df[f'dataframe_{j}_voxel_i'] = neuroimaging_df.iloc[:, i]
                variable_names.append(f'dataframe_{j}_voxel_i')
        
        if predictor_clinical_dfs[0] is not None:
            for j, clinical_df in enumerate(predictor_clinical_dfs):
                temp_df = temp_df.merge(clinical_df, left_index=True, right_index=True)
                variable_names.extend(clinical_df.columns.tolist())
            
        # Construct the regression formulas dynamically
        variables_combined = " + ".join(variable_names)
        interaction_terms = " + ".join([f"{var1}:{var2}" for idx, var1 in enumerate(variable_names) for var2 in variable_names[idx+1:]])
        
        formula_interaction = f'outcome ~ {variables_combined} + {interaction_terms}'
                
        # Fit the models
        if model_type == 'linear':
            model_interaction = smf.ols(formula_interaction, data=temp_df).fit()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if manual_t_stat:
            # Calculate the t-statistic manually
            print('under development')
        else:
            # Extract t-statistic from from statsmodels regression
            t_statistic = model_interaction.tvalues[-1] 
            if np.isinf(t_statistic):
                print("Warning: Infinite t-statistic detected. Setting p-value to zero.")
                P_value = 0.0
            else:
                P_value = model_interaction.pvalues[-1]
            statistic = 'statsmodels_t_statistic'
        # Store the results for the current voxel
        if comprehensive_results:
            voxel_results = {
                'voxel_index': i,
                'statistic': t_statistic,
                'statistic_method': statistic,
                'one_way_p_value': P_value,
                'unc_r_squared': model_interaction.rsquared,
                'adj_r_squared': model_interaction.rsquared_adj
            }
        else:
            voxel_results = {
                'voxel_index': i,
                'statistic': t_statistic
            }
        # Append the voxel_results dictionary to the results list
        results.append(voxel_results)

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    print('Example formula with interaction: ', formula_interaction)
    if permutation:
        return results_df['statistic']
    else:
        return results_df['statistic'], results_df, temp_df
#-----DEVELOPMENT --------------------------------

def generate_r_map(matrix_df, mask_path=None, method='pearson', tqdm_on=True):
    '''
    This function receive a dataframe which contains clinical outcomes in the first column, 
    and connectivity values of voxels in the proceeding columns.
    Each row of the dataframe represents a given patient
    
    Args:
    matrix_df: matrix to be processed
    mask_path: the flattned nifti data matrix of the mask to use for R-map generation
    
    Returns:
    r_map_df
    p_map_df
    r_squared_map_df
    '''
    #Mask the dataframe to save on computational time
    #Isolate the clinical variables
    outcomes_df = matrix_df.iloc[:,0]
    #Isolate voxels of interest
    matrix_df = mask_matrix(matrix_df, mask_path=mask_path, mask_by='columns')

    r_list = []
    p_list = []
    loop_range = tqdm(range(matrix_df.shape[1])) if tqdm_on else range(matrix_df.shape[1])
    for i in loop_range:
        if method=='pearson':
            r, p = pearsonr(outcomes_df, matrix_df.iloc[:,i])
        elif method=='spearman':
            r, p = spearmanr(outcomes_df, matrix_df.iloc[:,i])
        else:
            raise ValueError("Invalid method. Choose either 'spearman' or 'pearson'.")
            
        r_list.append(r)
        p_list.append(p)
        
    #Unmask the dataframe 
    r_map_df = pd.DataFrame(unmask_matrix(r_list, mask_path=mask_path))
    p_map_df = pd.DataFrame(unmask_matrix(p_list, mask_path=mask_path))
    r_squared_map_df = np.square(r_map_df)
    return r_map_df, p_map_df, r_squared_map_df
            
def generate_delta_r_map(matrix_df, threshold_of_interest, column_of_interest):
    '''
    This requires a dataframe with clinical outcomes in column 0, voxels in all proceeding columns, 
    and a named colum of interest which has values that can be used to split subgroups out.
    
    Args:
    matrix_df - the dataframe to generate delta_r_map from
    threshold_of_interest - the exact value to split subgroups by. 
    column_of_interest - the name of the column to split subgroups by.
    
    returns: delta_r_map
    '''
    # Find the indices of rows where the column value is below the threshold
    rows_under_threshold = matrix_df[matrix_df[column_of_interest] <= threshold_of_interest].index

    # Set the two different dataframes
    matrix_df.pop(column_of_interest)

    # #Initialize the dataframes
    dataframe_over_threshold = matrix_df.copy()
    dataframe_under_threshold = matrix_df.copy()

    #Set the specific rows to be included in each dataframe 
    rows_over_threshold = matrix_df.index.difference(rows_under_threshold)
    dataframe_over_threshold = matrix_df.loc[rows_over_threshold, :]
    dataframe_under_threshold = matrix_df.loc[rows_under_threshold, :]
    print('Dataframe over threshold is shape: ', dataframe_over_threshold.shape)
    print('Dataframe under threshold is shape: ', dataframe_under_threshold.shape)

    over_r_map_df, _, _ = generate_r_map(dataframe_over_threshold)
    under_r_map_df, _, _ = generate_r_map(dataframe_under_threshold)
    delta_r_map = over_r_map_df - under_r_map_df
    return delta_r_map

def permuted_patient_label_delta_r_map(dataframe_to_permute, observed_delta_r_map, column_of_interest, threshold_of_interest, n_permutations):
    '''
    Performs a permutation test on a given DataFrame using the observed delta_r_map and computes p-values.
    
    Args:
    dataframe_to_permute (pd.DataFrame): The input DataFrame to perform the permutation test on.
    observed_delta_r_map (pd.DataFrame): The observed delta_r_map to compare against permuted delta_r_maps.
    column_of_interest (str): The name of the column used for splitting subgroups.
    threshold_of_interest (float): The exact value used for splitting subgroups.
    n_permutations (int): The number of permutations to perform.
    
    Returns:
    pd.DataFrame: A DataFrame with the same shape as observed_delta_r_map, containing p-values.
    '''

    # Initialize a DataFrame with zeros and the same shape as observed_delta_r_map
    p_values_df = pd.DataFrame(np.zeros_like(observed_delta_r_map))

    # Loop through the number of permutations specified
    for i in range(0, n_permutations):
        # Make a copy of the input DataFrame
        dataframe_for_permutation = dataframe_to_permute.copy()

        # Permute the patient labels
        permuted_patient_labels = permute_column(dataframe_for_permutation.index.to_numpy())

        # Assign the permuted patient labels to the DataFrame's index
        dataframe_for_permutation.index = permuted_patient_labels
        patient_permuted_df = dataframe_for_permutation

        # Calculate the delta_r_map for the permuted DataFrame
        permuted_delta_r_map = generate_delta_r_map(patient_permuted_df, threshold_of_interest=threshold_of_interest, column_of_interest=column_of_interest)

        # Assess the significance of the permuted delta_r_map compared to the observed delta_r_map
        # If the observed delta_r_map is higher than the permuted delta_r_map, increment the corresponding p-value entry
        p_values_df = np.where(observed_delta_r_map > permuted_delta_r_map, p_values_df + 1, p_values_df)

    # Return the p_values_df
    return p_values_df