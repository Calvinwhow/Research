from calvin_utils.permutation_analysis_utils.palm_utils import CalvinPalm
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from tqdm import tqdm
import pandas as pd

class CalvinStatsmodelsPalm(CalvinPalm):
    """
    Subclass of CalvinPalm for handling PALM analysis with Statsmodels within a Jupyter Notebook.
    """
    
    def __init__(self, input_csv_path, output_dir, sheet=None):
        super().__init__(input_csv_path, output_dir, sheet=sheet)
        
    def read_and_display_data(self):
        """
        Reads data using the parent class method and displays the DataFrame.
        """
        data_df = super().read_data()
        display(data_df)
        return data_df
    
    def define_design_matrix(self, formula, data_df, voxelwise_variable=None):
        """
        Defines the design matrix based on the patsy formula and returns it as a DataFrame.
        
        Parameters:
        - formula: str, the patsy formula to construct the design matrix
        - data_df: DataFrame, the data frame from which to construct the design matrix
        - voxelwise_variablre: str, the column in data_df with paths to the voxelwise regressor files (niftis)
        
        Returns:
        - Tuple containing the design matrix for the dependent variable and the design matrix for the independent variables.
        """
        if voxelwise_variable is not None:
            vars = patsy.ModelDesc.from_formula(formula)
            vars.rhs_termlist.remove(patsy.Term([patsy.EvalFactor(voxelwise_variable)]))
            y, X = patsy.dmatrices(vars, data_df, return_type='dataframe')
            X[voxelwise_variable] = data_df[voxelwise_variable]
        else:
            y, X = patsy.dmatrices(formula, data_df, return_type='dataframe')
        return y, X

    def drop_nans_from_columns(self, columns_to_drop_from=None):
        """
        Drops rows with NaNs from specified columns in the DataFrame.

        Parameters:
        - columns_to_drop_from: list or None, columns to consider for NaN dropping.
        """
        if columns_to_drop_from is None:
            self.df.dropna(inplace=True)
        else:
            # Ensure that columns_to_drop_from is a list even if a single column name is provided
            if not isinstance(columns_to_drop_from, list):
                columns_to_drop_from = [columns_to_drop_from]
            self.df.dropna(subset=columns_to_drop_from, inplace=True)
        return self.df


    def drop_rows_based_on_value(self, column, condition, value):
        """
        Drops rows from the DataFrame based on a condition and value in a specified column,
        and returns the remaining DataFrame as well as a DataFrame of the dropped rows.

        Parameters:
        - column: str, the name of the column to evaluate
        - condition: str, the condition to evaluate ('equal', 'above', 'below')
        - value: numeric or str, the value to compare against

        Returns:
        - df: DataFrame, the DataFrame after dropping the rows
        - other_df: DataFrame, the DataFrame containing the dropped rows
        """
        if condition == 'equal':
            other_df = self.df[self.df[column] == value]
            self.df = self.df[self.df[column] != value]
        elif condition == 'above':
            other_df = self.df[self.df[column] > value]
            self.df = self.df[self.df[column] <= value]
        elif condition == 'below':
            other_df = self.df[self.df[column] < value]
            self.df = self.df[self.df[column] >= value]
        elif condition == 'not':
            other_df = self.df[self.df[column] != value]
            self.df = self.df[self.df[column] == value]
        else:
            raise ValueError(f"Condition '{condition}' is not supported.")
        
        return self.df, other_df


    def standardize_columns(self, cols_to_exclude=None):
        """
        Standardizes the columns of the DataFrame except those listed to exclude.

        Parameters:
        - cols_to_exclude: list, columns not to standardize
        """
        if cols_to_exclude is None:
            cols_to_exclude = []

        for col in self.df.columns:
            if col not in cols_to_exclude:
                try:
                    self.df[col] = (self.df[col] - np.mean(self.df[col])) / np.std(self.df[col])
                except Exception as e:
                    print(f'Unable to standardize column {col}')
        return self.df
    
    def run_mixed_effects_model(self, y, X, groups, random_intercepts=True, random_slopes=None):
        """
        Fits a mixed effects model given the design matrices and groups.
        
        Parameters:
        - y: DataFrame, the outcome (dependent variable) design matrix
        - X: DataFrame, the predictor (independent variables) design matrix
        - groups: Series or array-like, the grouping variable for random effects
        - random_intercepts: bool, whether to include random intercepts
        - random_slopes: list, None or columns to include for random slopes
        
        Returns:
        - A fitted mixed effects model result
        """
        
        # Start with a full copy of X for the random effects structure
        exog_re = X.copy()
        
        # Start with a full copy of X for the random effects structure
        exog_re = X.copy() if random_intercepts else X.drop('Intercept', axis=1)
        
        # If random slopes are specified, retain only those columns
        if random_slopes is not None:
            exog_re = exog_re[random_slopes + ['Intercept']] if random_intercepts else exog_re[random_slopes]

        # Initialize and fit the mixed effects model
        model = sm.MixedLM(y, X, groups, exog_re=exog_re)
        result = model.fit()
        
        return result

class RegressionAnalysis:
    def __init__(self, outcome_df, design_df, groups_df, N=100, metric='difference', two_tail=False, out_dir=None):
        self.outcome_df = outcome_df
        self.design_df = design_df
        self.groups_df = groups_df
        self.N = N
        self.metric = metric
        self.two_tail = two_tail
        self.out_dir = out_dir

    def calculate_p_values(self, permuted_delta_t_df, observed_delta_t_df, metric='difference', two_tail=True):
        # Get the unique column names from the observed_delta_t_df DataFrame
        columns = observed_delta_t_df.columns
        
        # Initialize a dictionary to store p-values for each regressor
        p_values_dict = {}
        
        if two_tail:
            permuted_delta_t_df = permuted_delta_t_df.abs()
            observed_delta_t_df = observed_delta_t_df.abs()
            
        # Calculate p-values based on the specified metric
        for column in columns:
            observed_t = observed_delta_t_df[column][0]
            permuted_t_values = permuted_delta_t_df[column]
            
            if metric == 'difference':
                p_value = (permuted_t_values > observed_t).mean()
            elif metric == 'similarity':
                p_value = (permuted_t_values < observed_t).mean()
            else:
                raise ValueError("Invalid metric. Use 'difference' or 'similarity'.")
            
            # Store the p-value in the dictionary with the regressor name as the key
            p_values_dict[column] = p_value
        
        # Create a DataFrame from the p-values dictionary
        p_values_df = pd.DataFrame.from_dict(p_values_dict, orient='index', columns=['p-value'])
        
        return p_values_df.transpose(), permuted_delta_t_df, observed_delta_t_df


    def delta_t(self, group_results, groups_df):
        # Get the unique values from the specified column in groups_df
        unique_groups = groups_df[groups_df.columns[0]].unique()
        
        # Get the number of permutations
        num_permutations = len(group_results[unique_groups[0]]['permuted_t_values'])
        
        # Get the observed t-values for each regressor
        observed_t_values = group_results[unique_groups[0]]['observed_t_values']
        
        # Initialize DataFrames to store permuted delta t and observed delta t
        permuted_delta_t_df = pd.DataFrame()
        observed_delta_t_df = pd.DataFrame(columns=observed_t_values.keys())
        
        # Calculate the permuted delta t-values for each regressor and permutation
        for regressor_name, _ in observed_t_values.items():
            # Initialize an array to store permuted delta t-values for this regressor and permutation
            permuted_delta_t_values = np.zeros(num_permutations)
            
            # Calculate the permuted delta t-values for this regressor and permutation
            for i in range(num_permutations):
                permuted_t_values_group1 = group_results[unique_groups[0]]['permuted_t_values'][i]
                permuted_t_values_group2 = group_results[unique_groups[1]]['permuted_t_values'][i]
                permuted_delta_t_values[i] = permuted_t_values_group1[regressor_name] - permuted_t_values_group2[regressor_name]
            
            # Store permuted delta t values in the DataFrame
            permuted_delta_t_df[regressor_name] = permuted_delta_t_values
        
        # Create DataFrames from the permuted delta t and observed delta t
        permuted_delta_t_df = pd.DataFrame(permuted_delta_t_df)
        observed_delta_t_df = pd.DataFrame([observed_t_values.values], columns=observed_t_values.keys())
        
        return permuted_delta_t_df, observed_delta_t_df
    
    def perform_permuted_regression(self, outcome_df, design_df, N=100):
        # Reset the indices of the DataFrames
        outcome_df = outcome_df.reset_index(drop=True)
        design_df = design_df.reset_index(drop=True)
        
        # Fit the regression model for the unpermuted data
        model = sm.OLS(outcome_df, design_df)
        results = model.fit()
        
        # Get the observed t-values
        observed_t_values = results.tvalues
        
        # Create a list to store permuted t-values for each run
        permuted_t_values = []
        
        # Perform permutation N times
        for _ in tqdm(range(N)):
            # Permute the outcome data
            permuted_outcome_df = outcome_df.copy()
            permuted_outcome_df = permuted_outcome_df.sample(frac=1).reset_index(drop=True)
            
            # Fit regression model for the permuted data
            model_permuted = sm.OLS(permuted_outcome_df, design_df)
            results_permuted = model_permuted.fit()
            
            # Get the permuted t-values
            permuted_t_values.append(results_permuted.tvalues)
        
        return observed_t_values, permuted_t_values

    def perform_permuted_regression_grouped(self, outcome_df, design_df, groups_df):
        # Initialize a dictionary to store results for each group
        group_results = {}
        
        # Get the unique values in the group column
        unique_groups = groups_df[groups_df.columns[0]].unique()
        
        # Iterate through unique groups
        for group_value in tqdm(unique_groups):
            # Get the indices of the current group
            group_indices = groups_df[groups_df[groups_df.columns[0]] == group_value].index
            
            # Subset the outcome and design DataFrames for the current group
            group_outcome_df = outcome_df.loc[group_indices]
            group_design_df = design_df.loc[group_indices]
            
            # Perform permuted regression for the current group
            t_obs, t_perm = self.perform_permuted_regression(group_outcome_df, group_design_df, N=self.N)
            
            # Store the results in the dictionary
            group_results[group_value] = {'observed_t_values': t_obs, 'permuted_t_values': t_perm}
        
        return group_results

    def plot_histogram_for_each_column(self, observed_delta_t_p, permuted_delta_t_df, observed_delta_t_df, bins=50, color_palette='dark'):
        num_columns = observed_delta_t_p.shape[1]
        fig, axes = plt.subplots(1, num_columns, figsize=(15, 5))
        sns.set_palette(color_palette)
        
        for i, column in enumerate(observed_delta_t_p.columns):
            observed_delta_t = observed_delta_t_df[column].values[0]
            permuted_delta_t = permuted_delta_t_df[column].values
            p_value = observed_delta_t_p[column].values[0]
            
            ax = axes[i]
            sns.histplot(permuted_delta_t, bins=bins, kde=True, label="Empirical $\\Delta t$ Distribution", element="step", color='blue', alpha=0.3, ax=ax)
            ax.axvline(x=observed_delta_t, color='red', linestyle='-', linewidth=1.5, label=f"Observed $\\Delta t$", alpha=0.6)
            ax.set_title(f"{column}\n$\\Delta t$ = {observed_delta_t:.2f}, p = {p_value:.4f}")
            ax.set_xlabel("$\\Delta t$")
            ax.set_ylabel("Count")
            
        plt.tight_layout()
        
        # Save the figure
        fig.savefig(f"{self.out_dir}/hist_kde_delta_t.png", bbox_inches='tight')
        fig.savefig(f"{self.out_dir}/hist_kde_delta_t.svg", bbox_inches='tight')
        print(f'Saved to {self.out_dir}/hist_kde_delta_t.svg')
        
    def run(self):
        group_results = self.perform_permuted_regression_grouped(outcome_df=self.outcome_df, design_df=self.design_df, groups_df=self.groups_df)
        permuted_delta_t_df, observed_delta_t_df = self.delta_t(group_results=group_results, groups_df=self.groups_df)
        observed_delta_t_p, permuted_delta_t_df, observed_delta_t_df = self.calculate_p_values(permuted_delta_t_df, observed_delta_t_df, metric=self.metric, two_tail=self.two_tail)
        self.plot_histogram_for_each_column(observed_delta_t_p, permuted_delta_t_df, observed_delta_t_df, bins=50, color_palette='dark')

class BootstrappedRegressionAnalysis(RegressionAnalysis):
    def __init__(self, outcome_df, design_df, groups_df, N=100, metric='difference', two_tail=False, out_dir=None, plot_together=True):
        super().__init__(outcome_df, design_df, groups_df, N, metric, two_tail, out_dir)
        self.plot_together = plot_together
    
    def perform_bootstrapped_regression(self, outcome_df, design_df, N=100):
        # Reset the indices of the DataFrames
        outcome_df = outcome_df.reset_index(drop=True)
        design_df = design_df.reset_index(drop=True)
        
        # Fit the regression model for the unpermuted data
        model = sm.OLS(outcome_df, design_df)
        results = model.fit()
        
        # Get the observed t-values
        observed_t_values = results.tvalues
        
        # Create a list to store bootstrap t-values for each run
        bootstrap_t_values = []
        
        # Perform bootstrap resampling N times
        for _ in tqdm(range(N)):
            # Resample the data with replacement
            sampled_indices = np.random.choice(outcome_df.index, len(outcome_df), replace=True)
            sampled_outcome_df = outcome_df.loc[sampled_indices].reset_index(drop=True)
            sampled_design_df = design_df.loc[sampled_indices].reset_index(drop=True)  # Ensure the design_df is also resampled accordingly
            
            # Fit regression model for the resampled data
            model_resampled = sm.OLS(sampled_outcome_df, sampled_design_df)
            results_resampled = model_resampled.fit()
            
            # Get the t-values for the resampled data
            bootstrap_t_values.append(results_resampled.tvalues)
        
        return observed_t_values, bootstrap_t_values

    
    def perform_bootstrapped_regression_grouped(self, outcome_df, design_df, groups_df):
        # Initialize a dictionary to store results for each group
        group_results = {}

        # Get the unique values in the group column
        unique_groups = groups_df[groups_df.columns[0]].unique()

        # Iterate through unique groups
        for group_value in tqdm(unique_groups):
            # Get the indices of the current group
            group_indices = groups_df[groups_df[groups_df.columns[0]] == group_value].index

            # Subset the outcome and design DataFrames for the current group
            group_outcome_df = outcome_df.loc[group_indices]
            group_design_df = design_df.loc[group_indices]

            # Perform bootstrapped regression for the current group
            t_obs, t_bootstrap = self.perform_bootstrapped_regression(group_outcome_df, group_design_df, N=self.N)

            # Store the results in the dictionary
            group_results[group_value] = {'observed_t_values': t_obs, 'bootstrap_t_values': t_bootstrap}

        return group_results
    
    def compile_bootstrapped_t_values(self, group_results, groups_df):
        # Get the unique group identifiers
        unique_groups = groups_df[groups_df.columns[0]].unique()

        # Create a dictionary to store the DataFrames for each group
        group_dataframes = {}

        # Iterate over the groups to create a DataFrame for each
        for group in unique_groups:
            # Get the number of bootstrap iterations
            num_bootstraps = len(group_results[group]['bootstrap_t_values'])

            # Create a list to hold the bootstrapped t-values for each iteration
            bootstrap_t_values = []

            # Extract the bootstrapped t-values
            for i in range(num_bootstraps):
                bootstrap_t_values.append(group_results[group]['bootstrap_t_values'][i])

            # Convert the list of bootstrapped t-values into a DataFrame
            group_dataframes[group] = pd.DataFrame(bootstrap_t_values)

        return group_dataframes
    
    def plot_violin_plot_for_each_group(self, bootstrap_dfs, alpha=0.05, color_palette='tab10'):
        # Determine the number of columns (regressors) from the first group's DataFrame
        num_columns = len(bootstrap_dfs[list(bootstrap_dfs.keys())[0]].columns)

        # Set up the figure for plotting
        fig, axs = plt.subplots(num_columns, 1, figsize=(3 * len(bootstrap_dfs.keys()), 5 * num_columns))
        
        # If there's only one column, wrap the ax in a list for consistent indexing
        if num_columns == 1:
            axs = [axs]

        sns.set_palette(color_palette)
        
        # Iterate through each column (regressor) to create a subplot
        for i, column in enumerate(bootstrap_dfs[list(bootstrap_dfs.keys())[0]].columns):
            data = []
            labels = []
            
            # Get bootstrap values for each group and add to the data list
            for group, df in bootstrap_dfs.items():
                bootstrap_t = df[column].values
                data.append(bootstrap_t)
                labels.append(f"Group {group}\n{column}")

            # Create violin plot for the current regressor
            sns.violinplot(data=data, ax=axs[i], inner="box", palette=color_palette, fill=False)
            axs[i].set_xticklabels(labels)
            axs[i].set_ylabel("Bootstrap T-value")
            
            # Calculate confidence intervals and observed T-values
            for j, (group, df) in enumerate(bootstrap_dfs.items()):
                # Calculate confidence intervals for the bootstrap T-values
                lower_ci = np.percentile(df[column], 100 * alpha / 2)
                upper_ci = np.percentile(df[column], 100 * (1 - alpha / 2))
                # Red line for observed T-value should be added here if available
                # axs[i].axhline(y=lower_ci, color='green', linestyle='--', alpha=0.6)
                # axs[i].axhline(y=upper_ci, color='green', linestyle='--', alpha=0.6)

        plt.tight_layout()

        # Save the figure
        out_dir = self.out_dir  # Assuming self.out_dir is set correctly
        fig.savefig(f"{out_dir}/violin_plot_t_values.png", bbox_inches='tight')
        fig.savefig(f"{out_dir}/violin_plot_t_values.svg", bbox_inches='tight')
        print(f'Saved to {out_dir}/violin_plot_t_values.svg')
        
    def plot_violin_plot_for_all_group(self, bootstrap_dfs, alpha=0.05, color_palette='dark'):
        # Assuming bootstrap_dfs is a dictionary of dataframes with group keys and t-values as columns
        
        # Prepare a combined DataFrame for plotting
        combined_df = pd.DataFrame()
        for group, df in bootstrap_dfs.items():
            df_long = pd.melt(df, var_name='Regressor', value_name='Bootstrap T-value')
            df_long['Group'] = group
            combined_df = pd.concat([combined_df, df_long], ignore_index=True)
        
        # Set up the figure for plotting
        plt.figure(figsize=(10, 5))
        sns.set_palette(color_palette)
        
        # Create violin plot for all regressors split by group
        sns.violinplot(
            data=combined_df,
            x='Regressor', y='Bootstrap T-value', hue='Group', split=True, gap=.3,
            inner="quart", palette=color_palette
        )

        plt.legend(title='Group')
        plt.tight_layout()
        
        # Save the figure
        out_dir = self.out_dir # Replace with your output directory
        plt.savefig(f"{out_dir}/combined_violin_plot_t_values.png", bbox_inches='tight')
        plt.savefig(f"{out_dir}/combined_violin_plot_t_values.svg", bbox_inches='tight')
        print(f'Saved to {out_dir}/combined_violin_plot_t_values.svg')

    def run(self):
        group_results = self.perform_bootstrapped_regression_grouped(outcome_df=self.outcome_df, design_df=self.design_df, groups_df=self.groups_df)
        bootstrap_df = self.compile_bootstrapped_t_values(group_results, self.groups_df)
        if self.plot_together:
            self.plot_violin_plot_for_all_group(bootstrap_df, alpha=0.05, color_palette='tab10')
        else:
            self.plot_violin_plot_for_each_group(bootstrap_df, alpha=0.05, color_palette='tab10')
            
        return bootstrap_df