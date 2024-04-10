from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ranksums
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

def get_sem(group):
    return group.sem()

class RCTPlotter:
    def __init__(self, data, obs_cols, arm_col, category_col=None, out_dir=None):
        """
        Initialize the RCTPlotter class.
        
        Args:
            data (pandas.DataFrame): DataFrame containing the data.
            obs_cols (list): List of column names containing the observations for each timepoint.
            arm_col (str): Name of the column containing the arm labels.
            category_col (str, optional): Name of the column containing the category/cohort labels.
        """
        self.data = data
        self.obs_cols = obs_cols
        self.arm_col = arm_col
        self.category_col = category_col
        self.out_dir = out_dir
        
        if category_col is None:
            self.categories = ['All']
        else:
            self.categories = self.data[category_col].unique()
        
        self.arms = self.data[arm_col].unique()

    def plot_rct_timecourse(self, data=None):
        color_palette = sns.color_palette("husl", len(self.categories) * len(self.arms))
        timepoints = range(1, len(self.obs_cols) + 1)
        
        for i, category in enumerate(self.categories):
            fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure and axes for each category

            if data is not None:
                category_data = data if self.category_col is None else data[data[self.category_col] == category]
            else:
                category_data = self.data if self.category_col is None else self.data[self.data[self.category_col] == category]
            
            for j, arm in enumerate(self.arms):
                arm_data = category_data[category_data[self.arm_col] == arm][self.obs_cols]
                means = arm_data.mean().values
                sems = arm_data.apply(get_sem).values  # Calculate SEM for each column and convert to numpy array
                label = f"{category} - {arm}" if self.category_col else arm
                color_idx = i * len(self.arms) + j
                # Plot error bars
                ax.errorbar(x=timepoints, y=means, yerr=sems, fmt='o', color=color_palette[color_idx], label=label)
                
                # Connect the points with lines
                ax.plot(timepoints, means, marker='', linestyle='-', color=color_palette[color_idx])

            ax.set_xlabel('Timepoints')
            ax.set_ylabel('Outcome Measure')
            ax.legend()
            if self.out_dir:
                plt.savefig(os.path.join(self.out_dir, f"{category}rct_plot.png"))
                plt.savefig(os.path.join(self.out_dir, f"{category}rasterized_probabilities.svg"))
            plt.show()  # Show the plot for the current category
            
    def perform_ttest(self):
        """
        Perform t-tests between the arms at each timepoint and print the results.
        """
        for timepoint_idx, obs_col in enumerate(self.obs_cols):
            print(f"Timepoint {timepoint_idx + 1}:")

            for category in self.categories:
                category_data = self.data if self.category_col is None else self.data[self.data[self.category_col] == category]

                # Ensure data is numeric before performing t-test
                for arm1 in self.arms:
                    for arm2 in self.arms:
                        if arm1 == arm2:
                            continue  # Skip comparing the same arms

                        arm1_data = category_data[category_data[self.arm_col] == arm1][obs_col].astype(float)
                        arm2_data = category_data[category_data[self.arm_col] == arm2][obs_col].astype(float)

                        # Perform t-test and handle cases with insufficient data
                        if len(arm1_data.dropna()) > 1 and len(arm2_data.dropna()) > 1:
                            t_stat, p_value = ttest_ind(arm1_data.values, arm2_data.values, equal_var=True)
                            welch_stat, wp_value = ttest_ind(arm1_data.values, arm2_data.values, equal_var=True)
                            u_stat, up_value = ranksums(arm1_data.values, arm2_data.values)
                            print(f"  {category} - {arm1} vs {arm2}: t-statistic={t_stat:.2f}, p-value={p_value:.4f}")
                            print(f"  {category} - {arm1} vs {arm2}: welch-statistic={welch_stat:.2f}, p-value={wp_value:.4f}")
                            print(f"  {category} - {arm1} vs {arm2}: u-statistic={u_stat:.2f}, p-value={up_value:.4f}")
                        else:
                            print(f"  {category} - {arm1} vs {arm2}: Insufficient data for t-test")

            print()  # Print an empty line to separate timepoints

    def run(self):
        """
        Run the RCTPlotter and display the plot.
        """
        ax = self.plot_rct_timecourse()
        self.perform_ttest()

class DiDAnalysis(RCTPlotter):
    def __init__(self, data, obs_cols, arm_col, category_col=None):
        super().__init__(data, obs_cols, arm_col, category_col)
        self.timepoints = range(1, len(self.obs_cols) + 1)

    def plot_did_timecourse(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        color_palette = sns.color_palette("husl", len(self.categories) * len(self.arms))
        
        for i, category in enumerate(self.categories):
            for j, arm in enumerate(self.arms):
                label = f"{category} - {arm}" if self.category_col else arm
                color_idx = i * len(self.arms) + j
                
                # Aggregate data for each arm within the category
                arm_data = self.data[(self.data[self.arm_col] == arm) & (self.data[self.category_col] == category) if self.category_col else (self.data[self.arm_col] == arm)]
                
                # Calculate means and SEMs for each timepoint
                means = arm_data[self.obs_cols].mean()
                sems = arm_data[self.obs_cols].sem()
                
                # Plot
                ax.errorbar(x=self.timepoints, y=means, yerr=sems, fmt='o-', color=color_palette[color_idx], label=label)
        
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Outcome Measure')
        ax.legend()

        return ax

    def run(self):
        ax = self.plot_did_timecourse()
        plt.show()

class PropensityStratifiedRCTPlotter(RCTPlotter):
    def __init__(self, data, obs_cols, arm_col, category_col=None, covariate_cols=None, n_strata=5):
        """
        Initialize the PropensityStratifiedRCTPlotter class.
        
        Args:
            data (pandas.DataFrame): DataFrame containing the data.
            obs_cols (list): List of column names containing the observations for each timepoint.
            arm_col (str): Name of the column containing the arm labels.
            category_col (str, optional): Name of the column containing the category/cohort labels.
            covariate_cols (list, optional): List of column names to be used as covariates in the propensity score model.
            n_strata (int, optional): Number of strata to divide the data into based on propensity scores.
        """
        super().__init__(data, obs_cols, arm_col, category_col)
        self.covariate_cols = covariate_cols
        self.n_strata = n_strata

    def calculate_propensity_scores(self):
        """
        Calculate propensity scores using logistic regression based on specified covariates.
        """
        if self.covariate_cols is None:
            raise ValueError("Covariate columns must be specified for propensity score calculation.")

        # Prepare the data
        X = self.data[self.covariate_cols]
        y = self.data[self.arm_col].astype('category').cat.codes

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit logistic regression to estimate propensity scores
        logistic_model = LogisticRegression()
        logistic_model.fit(X_scaled, y)
        propensity_scores = logistic_model.predict_proba(X_scaled)[:, 1]

        # Add propensity scores to the DataFrame
        self.data['propensity_score'] = propensity_scores

    def stratify_data(self):
        """
        Stratify the data into specified number of strata based on propensity scores.
        """
        # Create strata based on propensity score quantiles
        self.data['stratum'] = pd.qcut(self.data['propensity_score'], self.n_strata, labels=False, duplicates='drop')
        
    def assess_effect_size(self):
        stratum_effects = []
        stratum_weights = []

        arm_categories = self.data[self.arm_col].unique()
        if len(arm_categories) != 2:
            raise ValueError("The arm column must have exactly two unique categories for comparison.")

        for stratum in sorted(self.data['stratum'].unique()):
            stratum_data = self.data[self.data['stratum'] == stratum]
            group1_data = stratum_data[stratum_data[self.arm_col] == arm_categories[0]][self.obs_cols].mean(axis=1)
            group2_data = stratum_data[stratum_data[self.arm_col] == arm_categories[1]][self.obs_cols].mean(axis=1)

            if group1_data.empty or group2_data.empty:
                print(f"Stratum {stratum} has insufficient data for one or both groups. Skipping.")
                continue

            # Calculate median difference in this stratum as an effect size measure
            median_diff = group1_data.median() - group2_data.median()

            # Perform the Wilcoxon rank-sum test
            stat, wp_value = ranksums(group1_data.dropna(), group2_data.dropna())
            t_stat, tp_value = ttest_ind(group1_data.dropna(), group2_data.dropna())
            print(t_stat, tp_value)

            stratum_effects.append(median_diff)
            stratum_weights.append(len(stratum_data))

            print(f"Stratum {stratum}: Median Difference = {median_diff:.4f}, p-value = {wp_value:.4f}")

        if stratum_effects and stratum_weights:
            weighted_avg_effect = np.average(stratum_effects, weights=stratum_weights)
            print(f"\nWeighted Average Effect Size: {weighted_avg_effect:.4f}")
        else:
            print("No valid strata were found for analysis.")

        return weighted_avg_effect

    def plot_rct_timecourse(self):
        """
        Override the plot_rct_timecourse method to plot within each propensity score stratum.
        """
        self.calculate_propensity_scores()
        self.stratify_data()
        self.assess_effect_size()

        for stratum in self.data['stratum'].unique():
            stratum_data = self.data[self.data['stratum'] == stratum]
            print(f"Plotting for stratum: {stratum}")
            super().plot_rct_timecourse(data=stratum_data)
            
    