import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import os 

class ModelFreeInteractionPlot:
    '''
    This class calculates the mean and standard error of the mean (SEM) for specified groupings in the data 
    and plots the mean +/- SEM across several categorical values. It generates an interaction plot without 
    using a statistical model.
    '''
    
    @staticmethod
    def calculate_means_and_sem(data_df, group1_column, group2_column, outcome_column):
        """
        Calculate means and standard errors of the mean (SEM) for specified groupings.

        Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        group1_column (str): Name of the first grouping column (categorical).
        group2_column (str): Name of the second grouping column (categorical).
        outcome_column (str): Name of the outcome column (continuous).

        Returns:
        pd.DataFrame: Summary DataFrame with columns for the group levels, mean, standard deviation, count, and SEM.

        The function groups the data by `group1_column` and `group2_column`, calculates the mean, standard deviation, 
        and count of the `outcome_column` for each group, and then computes the SEM for each group.
        """
        # Group data and calculate mean, standard deviation, and count of samples
        summary_df = data_df.groupby([group2_column, group1_column])[outcome_column].agg(['mean', 'std', 'count']).reset_index()
        summary_df.columns = [group2_column, group1_column, 'Average', 'STD', 'Count']

        # Calculate SEM only for groups with more than one data point
        summary_df['SEM'] = summary_df.apply(lambda row: row['STD'] / np.sqrt(row['Count']) if row['Count'] > 1 else np.nan, axis=1)

        # Drop any rows with NaN values in 'Average'
        summary_df.dropna(subset=['Average'], inplace=True)

        return summary_df

    @staticmethod
    def plot_interaction_error_bar(summary_df, group1_column, group2_column, out_dir=None):
        """
        Plot an interaction error bar plot showing the relationship between two factors with SEM.

        Args:
        summary_df (pd.DataFrame): Summary DataFrame containing the calculated means and SEM.
        group1_column (str): Name of the first grouping column (categorical).
        group2_column (str): Name of the second grouping column (categorical).
        out_dir (str, optional): Directory to save the plot. If None, the plot is not saved.

        The function generates an interaction plot with error bars representing the SEM. 
        The x-axis represents the levels of `group1_column`, the y-axis represents the mean outcome, 
        and different lines represent the levels of `group2_column`.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))

        # Unique categories in the second grouping variable
        group2_categories = summary_df[group2_column].unique()

        # Colors for each line
        colors = sns.color_palette("tab10", len(group2_categories))

        for idx, category in enumerate(group2_categories):
            # Subset for each category of group 2
            subset = summary_df[summary_df[group2_column] == category]
            # Plot points. If SEM is NaN, error bars are not drawn.
            plt.errorbar(subset[group1_column], subset['Average'], capsize=6, yerr=subset['SEM'], fmt='-o', label=category, color=colors[idx])

        plt.xlabel(group1_column)
        plt.ylabel('Average Outcome')
        plt.title('Interaction Plot of Average Outcomes with SEM')
        plt.legend(title=group2_column, loc='upper right')
        plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

        # Save the figure if a directory is provided
        if out_dir:
            plt.savefig(f"{out_dir}/interaction_means_error_bars.png", bbox_inches='tight')
            plt.savefig(f"{out_dir}/interaction_means_error_bars.svg", bbox_inches='tight')
            print(f'Saved to {out_dir}/interaction_means_error_bars.svg')
        
        # Show the plot
        plt.show()
        
    @staticmethod
    def diagnose_data(data_df, group1_column, group2_column, outcome_column):
        group2_categories = data_df[group2_column].unique()

        for category in group2_categories:
            subset = data_df[data_df[group2_column] == category]
            print(f"Category: {category}")
            groups = [subset[subset[group1_column] == group][outcome_column].values for group in subset[group1_column].unique()]
            
            for i, group in enumerate(subset[group1_column].unique()):
                print(f"  Group {group} size: {len(groups[i])}, values: {groups[i]}")
                
            if any(len(group) <= 1 for group in groups):
                print("  Issue: One or more groups have insufficient data points.")
            elif all(np.all(group == groups[0][0]) for group in groups):
                print("  Issue: All values in the groups are the same.")
                
    @staticmethod
    def perform_kruskal_wallis_test(data_df, group1_column, group2_column, outcome_column):
        """
        Performs the Kruskal-Wallis test to compare the distribution of outcome values across
        multiple groups defined by group1_column within each level of group2_column.

        Parameters:
        - data_df: DataFrame containing the data.
        - group1_column: String, the name of the first grouping column.
        - group2_column: String, the name of the second grouping column.
        - outcome_column: String, the name of the outcome column.

        Returns:
        - results: Dictionary containing the p-values of the Kruskal-Wallis test for each level of group2_column.
        """
        results = {}
        # Get unique categories in the second grouping variable
        group2_categories = data_df[group2_column].unique()

        for category in group2_categories:
            # Filter data for each category of group2
            subset = data_df[data_df[group2_column] == category]

            # Collect the data from each group in group1_column, removing NaN values
            groups = [subset[subset[group1_column] == group][outcome_column].dropna().values for group in subset[group1_column].unique()]

            # Check if all groups have data after removing NaNs
            if all(len(group) > 0 for group in groups):
                # Perform Kruskal-Wallis test
                statistic, p_value = kruskal(*groups)
                results[category] = p_value
            else:
                results[category] = 'Not enough data to perform test'

        return results

    @staticmethod
    def perform_contrast(data_df, contrast_column, outcome_column):
        """
        Compare data across all cohorts between two levels using the Mann-Whitney U test.

        Parameters:
        - data_df (pd.DataFrame): DataFrame containing the data.
        - contrast_column (str): The column that specifies the subgroup levels, which must contain exactly two unique values.
        - outcome_column (str): The column containing the data to compare (continuous).

        Returns:
        - result (dict): A dictionary containing the U statistic and p-value of the test, or an error message if there is insufficient data.
        """
        # Ensure there are exactly two unique values in the contrast column
        unique_values = data_df[contrast_column].unique()
        if len(unique_values) != 2:
            return {'Error': 'Contrast column must have exactly two unique values'}

        # Assign the unique values to low and high
        low_value, high_value = unique_values

        # Filter data into two groups based on the levels
        low_data = data_df[data_df[contrast_column] == low_value][outcome_column].dropna()
        high_data = data_df[data_df[contrast_column] == high_value][outcome_column].dropna()

        # Perform the Mann-Whitney U test
        if len(low_data) > 0 and len(high_data) > 0:
            statistic, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')
            return {'U statistic': statistic, 'p-value': p_value}
        else:
            return {'Error': 'Insufficient data'}

class QuickANOVAPlot:
    """
    QuickANOVAPlot is a utility class for generating interaction plots from a fitted model and design matrix.

    Parameters:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model object from statsmodels.
    design_matrix : pandas.DataFrame or numpy.ndarray
        The design matrix used for the model. This should include all predictor variables used in the model.
    out_dir : str, optional
        Directory to save the plot. If None, the plot will not be saved.

    Methods:
    --------
    make_predictions():
        Generates predictions from the model using the design matrix.
    
    plot_predictions(x_var, hue_var, cohort_var, y_label='Predictions', title='Interaction Plot of Predictions'):
        Plots the interaction effects using the specified variables.

    Parameters for plot_predictions:
    --------------------------------
    x_var : str
        The variable to be plotted on the x-axis.
    hue_var : str
        The variable to differentiate lines within the plot.
    cohort_var : str, optional
        The variable to differentiate cohorts with different colors. If None, cohorts are not differentiated.
    y_label : str, optional
        Label for the y-axis. Default is 'Predictions'.
    title : str, optional
        Title for the plot. Default is 'Interaction Plot of Predictions'.

    Notes:
    ------
    - The class expects the design matrix to contain both categorical and continuous variables.
    - It can handle models with more than two or three variables, but the plot will only display the interaction between the specified variables.
    - Categorical variables should be properly encoded (e.g., using pandas' categorical dtype or similar).
    - The class is flexible and can generalize to different models, provided the necessary variables are specified for plotting.
    """

    def __init__(self, model, design_matrix, out_dir=None):
        self.model = model
        self.design_matrix = design_matrix
        self.predictions = None
        self.out_dir = out_dir

    def make_predictions(self):
        """Generates predictions from the model using the design matrix."""
        self.predictions = self.model.predict(self.design_matrix)

    def plot_predictions(self, x_var, hue_var, cohort_var=None, y='predictions', error_bar='ci'):
        """
        Plots the interaction effects using the specified variables.

        Parameters:
        -----------
        x_var : str
            The variable to be plotted on the x-axis.
        hue_var : str
            The variable to differentiate lines within the plot.
        cohort_var : str, optional
            The variable to differentiate cohorts with different colors. If None, cohorts are not differentiated.
        y : str, optional
            The variable to plot on the y-axis. If 'predictions' (default), will plot the model's predictions (Estimated Marginal Means).
        error_bar : str, optional
            The size of the confidence interval to draw when aggregating with an estimator. Default is 'ci'.
            options: 'ci' | 'sd' | 'se' | None
        """
        if self.predictions is None:
            raise ValueError("Predictions have not been made yet. Call make_predictions() first.")

        # Convert the design matrix to a DataFrame if it isn't already
        if not isinstance(self.design_matrix, pd.DataFrame):
            self.design_matrix = pd.DataFrame(self.design_matrix)

        # Add predictions to the design matrix DataFrame
        self.design_matrix['predictions'] = self.predictions

        # Ensure variables are in the design matrix
        if x_var not in self.design_matrix.columns or hue_var not in self.design_matrix.columns or (cohort_var and cohort_var not in self.design_matrix.columns):
            raise ValueError("Variables for x, hue, or cohort are not in the design matrix.")
        # Plot the interaction plot with traces for cohorts
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 8))
        y_label=y.capitalize()
        title=f'Plot of {y_label}'

        # Iterate through each cohort and plot
        if cohort_var is not None:
            cohort_colors = sns.color_palette("tab10", len(self.design_matrix[cohort_var].unique()))
            for i, cohort in enumerate(self.design_matrix[cohort_var].unique()):
                cohort_data = self.design_matrix[self.design_matrix[cohort_var] == cohort]

                # Plot with solid and dashed lines
                for j, hue in enumerate(cohort_data[hue_var].unique()):
                    hue_data = cohort_data[cohort_data[hue_var] == hue]
                    sns.lineplot(
                        x=x_var, 
                        y=y, 
                        data=hue_data,
                        color=cohort_colors[i],
                        linestyle='--' if j % 2 == 0 else '-',
                        marker='o',
                        label=f'{cohort} ({hue})',
                        errorbar=error_bar
                        )
        else:
            # Plot without cohort differentiation
            sns.lineplot(
                x=x_var, 
                y=y, 
                hue=hue_var, 
                data=self.design_matrix,
                palette='tab10',
                style=hue_var,
                markers=True,
                dashes=[(10, 10), ''],  # Dashed line for first level, solid line for second level
                errorbar=error_bar
            )

        plt.xlabel(x_var)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(title=cohort_var, loc='upper right')
        sns.despine()
        plt.grid(False)
        if self.out_dir is not None:
            plt.savefig(os.path.join(self.out_dir, 'anova_plot.svg'))
        plt.show()
        
    def plot_bar(self, x_var, hue_var, cohort_var=None, y='predictions', y_label='Predictions', title='Bar Plot of Predictions', error_bar='ci'):
        """
        Creates a bar plot using the specified variables.

        Parameters:
        -----------
        x_var : str
            The variable to be plotted on the x-axis.
        hue_var : str
            The variable to differentiate bars within the plot.
        cohort_var : str, optional
            The variable to differentiate cohorts with different colors. If None, cohorts are not differentiated.
        y : str, optional
            The variable to plot on the y-axis. If 'predictions' (default), will plot the model's predictions.
        y_label : str, optional
            Label for the y-axis. Default is 'Predictions'.
        title : str, optional
            Title for the plot. Default is 'Bar Plot of Predictions'.
        error_bar : str, optional
            The size of the confidence interval to draw when aggregating with an estimator. Default is 'ci'.
            options: 'ci' | 'sd' | 'se' | None
        """
        if self.predictions is None:
            raise ValueError("Predictions have not been made yet. Call make_predictions() first.")

        # Convert the design matrix to a DataFrame if it isn't already
        if not isinstance(self.design_matrix, pd.DataFrame):
            self.design_matrix = pd.DataFrame(self.design_matrix)

        # Add predictions to the design matrix DataFrame
        self.design_matrix['predictions'] = self.predictions

        # Ensure variables are in the design matrix
        if x_var not in self.design_matrix.columns or hue_var not in self.design_matrix.columns or (cohort_var and cohort_var not in self.design_matrix.columns):
            raise ValueError("Variables for x, hue, or cohort are not in the design matrix.")

        # Plot the bar plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 8))
        y_label = y_label.capitalize()
        title = title

        # Bar plot with hue and cohort differentiation
        if cohort_var is not None:
            sns.barplot(
                x=x_var, 
                y=y, 
                hue=hue_var, 
                data=self.design_matrix,
                errorbar=error_bar,
                palette='tab10'
            )
            plt.legend(title=cohort_var, loc='upper right')
        else:
            sns.barplot(
                x=x_var, 
                y=y, 
                hue=hue_var, 
                data=self.design_matrix,
                errorbar=error_bar,
                palette='tab10'
            )

        plt.xlabel(x_var)
        plt.ylabel(y_label)
        plt.title(title)
        sns.despine()
        plt.grid(False)
        if self.out_dir is not None:
            plt.savefig(os.path.join(self.out_dir, 'bar_plot.svg'))
        plt.show()