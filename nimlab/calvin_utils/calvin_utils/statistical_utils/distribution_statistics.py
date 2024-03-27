import pandas as pd
import numpy as np
from scipy.stats import t
import numpy as np
from tqdm import tqdm
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class BootstrappedDistributionStatistics:
    """
    Class for calculating statistics from empirical data distributions.

    Attributes:
        data (array): The empirical data distribution.
        mean (float): The mean of the data.
        std_dev (float): The standard deviation of the data.
        sem (float): The standard error of the mean of the data.

    Methods:
        percentile_ci(alpha): Computes the percentile-based confidence interval.
        analytic_ci(alpha): Computes the analytic confidence interval, based on a t-distribution.
    """
    def __init__(self, data):
        """
        Initializes DistributionValues with data and calculates mean, standard deviation, and standard error of mean.
        
        Parameters:
            data (array-like): The empirical data distribution.
        """
        self.data = np.array(data)
        self.mean = np.mean(self.data)
        self.std_dev = np.std(self.data, ddof=1) #<--this is also called the 'standard error of the distribution'

    def percentile_ci(self, alpha=0.05):
        """
        Computes the percentile-based confidence interval.
        
        Parameters:
            alpha (float): Significance level. Default is 0.05.

        Returns:
            lower_bound (float): The lower bound of the confidence interval.
            upper_bound (float): The upper bound of the confidence interval.
        """
        lower_bound = np.percentile(self.data, 100 * alpha / 2.)
        upper_bound = np.percentile(self.data, 100 * (1 - alpha / 2.))
        return lower_bound, upper_bound

    def analytic_ci(self, alpha=0.05):
        """
        Computes the analytic confidence interval, based on a t-distribution.

        Parameters:
            alpha (float): Significance level. Default is 0.05.

        Returns:
            lower_bound (float): The lower bound of the confidence interval.
            upper_bound (float): The upper bound of the confidence interval.
        """
        df = len(self.data) - 1  # Degrees of freedom
        cv = t.ppf(1.0 - alpha / 2., df)  # Critical value of the t-distribution
        lower_bound = self.mean - cv * self.sem
        upper_bound = self.mean + cv * self.sem
        return lower_bound, upper_bound
    
    def compute_non_binary_sem(self):
        """
        Computes the standard error of the mean as is typically done.
        
        Returns:
            sem (float): The standard error of the mean.
        """
        return self.std_dev / np.sqrt(len(self.data))  # standard error of the mean

    
    def compute_binary_sem(self):
        """
        Computes the standard error of the mean for a Bernoulli distribution.
        This formula assumes that your data consists of 1's and 0's, representing 
        "success" and "failure" respectively. Note that this method of calculating 
        the standard error is specifically for proportions, such as binary/binary 
        classification outcomes. It may not be applicable for other types of data distributions.
        Returns:
            sem (float): The standard error of the mean.
        """
        return np.sqrt(self.mean * (1 - self.mean) / len(self.data))


#----------------------------------------------------------------MISC FUNCTIONS----------------------------------------------------------------

def calculate_distribution_confidence_intervals(distribution):
    """
    Calculates the confidence intervals and p-value based on the bootstrapped samples.

    Parameters:
    - ab_paths: list of lists containing the bootstrapped ab paths for each mediator.
    - total_indirect_effects: list of bootstrapped summed ab paths.
    - mediators: list of mediator names.

    Returns:
    - DataFrame with the mean indirect effect, confidence intervals, and p-values for each mediator and the total indirect effect.
    """

    ab_path_values = np.array(distribution)

    # Calculate mean indirect effect and confidence intervals for each mediator
    mean_ab_paths = np.mean(ab_path_values, axis=0)
    lower_bounds = np.percentile(ab_path_values, 2.5, axis=0)
    upper_bounds = np.percentile(ab_path_values, 97.5, axis=0)

    # Calculate p-values for each mediator
    ab_path_p_values = [np.mean(np.sign(mean_ab_paths) * ab_path_values <= 0)]

    # Create DataFrame to store the results
    result_df = pd.DataFrame({
        'Point Estimate': mean_ab_paths,
        '2.5th Percentile': lower_bounds,
        '97.5th Percentile': upper_bounds,
        'P-value': ab_path_p_values
    }, index=['Bootstrapped 95-CIs'])

    return result_df

def bootstrap_distribution_statistics(data, func, func_args=None, bootstrap_samples=2500):
    """
    Bootstraps the given data using the provided function.

    Parameters:
    - data: DataFrame with the data.
    - func: function to apply to each bootstrap sample.
    - func_args: dictionary of additional arguments to pass to func.
    - bootstrap_samples: number of bootstrap samples.

    Returns:
    - DataFrame with the point estimate, confidence intervals, p-value and standard error of mean.
    """

    # Initialize a list to store the bootstrap results
    bootstrap_results = []

    # Loop over each bootstrap sample
    for _ in tqdm(range(bootstrap_samples)):
        # Resample the data with replacement
        sample = data.sample(frac=1, replace=True)
        # Apply the function to the sample and append the result
        bootstrap_results.append(func(sample, **func_args))
    # Convert the results to a numpy array
    bootstrap_results = np.array(bootstrap_results)

    # Calculate mean, confidence intervals, p-value and standard error of the mean
    point_estimate = np.mean(bootstrap_results)
    lower_bound = np.percentile(bootstrap_results, 2.5)
    upper_bound = np.percentile(bootstrap_results, 97.5)
    p_value = np.mean(np.sign(point_estimate) * bootstrap_results <= 0)

    standard_errors = np.std(bootstrap_results, ddof=1) / np.sqrt(len(bootstrap_results))
    bootstrap_sem = np.mean(standard_errors)

    # Create a DataFrame to store the results
    result_df = pd.DataFrame({
        'Point Estimate': point_estimate,
        '2.5th Percentile': lower_bound,
        '97.5th Percentile': upper_bound,
        'P-value': p_value,
        'Standard Error of Mean': bootstrap_sem
    }, index=['Bootstrapped 95-CIs'])

    return result_df


def plot_with_annotation(dataframe, col1, col2, xlabel, ylabel, test_type, colours):
    # Validate the test type
    valid_tests = ['t-test_ind', 't-test_paired', 'Mann-Whitney', 'Wilcoxon']
    if test_type not in valid_tests:
        raise ValueError(f"Invalid test type. Choose from: {', '.join(valid_tests)}")
    
    # Extract series
    series1 = dataframe[col1]
    series2 = dataframe[col2]
    
    # Prepare data
    data = pd.DataFrame({'Group': [col1] * len(series1) + [col2] * len(series2),
                         'Value': series1.tolist() + series2.tolist()})
    
    # Pairs for comparison
    pairs = [(col1, col2)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    sns.set_palette(colours, 2, desat=1)
    
    # Plot with seaborn
    sns.boxplot(x='Group', y='Value', data=data, ax=ax)
    
    # Add annotations
    annotator = Annotator(ax=ax,
                          data=data,
                          x='Group',
                          y='Value',
                          pairs=pairs,
                          test=test_type,
                          text_format='full',
                          loc='inside',
                          verbose=2)
    
    # Configure and annotate
    _, corrected_results = annotator.configure(test=test_type, comparisons_correction="bonferroni").apply_and_annotate()
    
    # Label and show plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt

def invert_distribution(values):
    """
    Transform a distribution of values by inverting or normalizing them.

    Parameters:
    - values (numpy.array): A numpy array containing the raw values to be transformed.
    - invert (bool, optional): A boolean flag that determines the type of transformation. 
      If True, the distribution is inverted. If False, the distribution is normalized. 
      Defaults to False.

    Returns:
    - numpy.array: The transformed values.

    """

    # Invert the distribution
    min_value = np.min(values)
    max_value = np.max(values)
    transformed_values = max_value + min_value - values

    return transformed_values

def normalize_distribution(values):
    # Normalize the distribution
    min_value = np.min(values)
    max_value = np.max(values)
    transformed_values = (values - min_value) / (max_value - min_value)
    return transformed_values
