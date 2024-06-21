import pandas as pd
import numpy as np

def z_score_normalize(series):
    '''
    Function to Z-score normalize a series (standard score)
    '''
    mean = series.mean()
    std = series.std()
    return (series - mean) / std

def log_transform(series):
    '''
    Function to apply log transformation to a series
    '''
    return np.log(series + 1)  # Adding 1 to avoid log(0)

def sqrt_transform(series):
    '''
    Function to apply square root transformation to a series
    '''
    return np.sqrt(series)

def boxcox_transform(series, lmbda):
    '''
    Function to apply Box-Cox transformation to a series
    '''
    from scipy.stats import boxcox
    return boxcox(series + 1, lmbda)  # Adding 1 to avoid issues with non-positive values

def standardize_series(series):
    '''
    Function to standardize a series to have mean 0 and standard deviation 1
    '''
    return (series - series.mean()) / series.std()

def robust_scale(series):
    '''
    Function to scale a series using median and interquartile range (IQR)
    '''
    median = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)
    return (series - median) / iqr

def inverse_transform(series):
    '''
    Function to apply inverse transformation to a series
    '''
    return 1 / (series + 1)  # Adding 1 to avoid division by zero

def exp_transform(series):
    '''
    Function to apply exponential transformation to a series
    '''
    return np.exp(series)

def clip_outliers(series, lower_quantile=0.05, upper_quantile=0.95):
    '''
    Function to clip outliers in a series based on quantiles
    '''
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return series.clip(lower_bound, upper_bound)

def min_max_normalize_minus_one_to_one(series, reference_series=None):
    """
    Normalize a series to the range [-1, 1]. If a reference series is provided,
    use its min and max values for normalization; otherwise, use the series' own min and max values.

    Parameters:
    series (pd.Series): The series to be normalized.
    reference_series (pd.Series, optional): The reference series to use for normalization.

    Returns:
    pd.Series: The normalized series with values in the range [-1, 1].
    """
    if reference_series is not None:
        min_val = reference_series.min()
        max_val = reference_series.max()
        return 2 * (series - min_val) / (max_val - min_val) - 1
    else:
        min_val = series.min()
        max_val = series.max()
        return 2 * (series - min_val) / (max_val - min_val) - 1s

def min_max_normalize(series):
    '''
    Function to normalize a series to the range [0, 1]
    '''
    return (series - series.min()) / (series.max() - series.min())

def fisher_z_transform(series):
    '''
    Function to apply Fisher's Z-transformation to a series of correlation coefficients
    '''
    return np.arctanh(series)

def logit_transform(series):
    '''
    Function to apply logit transformation to a series
    '''
    return np.log(series / (1 - series))

def reciprocal_transform(series):
    '''
    Function to apply reciprocal transformation to a series
    '''
    return 1 / series

def yeo_johnson_transform(series):
    '''
    Function to apply Yeo-Johnson transformation to a series
    '''
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    return pd.Series(pt.fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)

def quantile_transform(series, n_quantiles=100):
    '''
    Function to apply quantile transformation to a series
    '''
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    return pd.Series(qt.fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)

def rank_transform(series):
    '''
    Function to apply rank transformation to a series
    '''
    return series.rank()

def square_transform(series):
    '''
    Function to apply square transformation to a series
    '''
    return np.square(series)

def cube_transform(series):
    '''
    Function to apply cube transformation to a series
    '''
    return np.power(series, 3)

def invert_distribution(series):
    """
    Invert the distribution of a series.

    Parameters:
    series (pd.Series): The series to be inverted.

    Returns:
    pd.Series: The series with its distribution inverted.
    """
    max_val = series.max()
    return max_val - series