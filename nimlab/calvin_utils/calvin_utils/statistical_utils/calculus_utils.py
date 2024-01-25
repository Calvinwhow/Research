# Redefining the function to incorporate the changes
import os
import numpy as np
import matplotlib.pyplot as plt

def find_zero_point_of_coefficients(statsmodel_model):
    """
    Calculate the values of independent variables at which the dependent variable
    is zero, based on the coefficients from a fitted statsmodels regression model.

    Parameters:
    statsmodel_model (RegressionResultsWrapper): A fitted statsmodels regression model.

    Returns:
    dict: A dictionary where keys are variable names and values are the calculated
          points at which the dependent variable is zero for each coefficient. 
          If a coefficient is zero, the corresponding value is None, as the
          zero point is undefined.
    """
    # Extracting the coefficients
    coefficients = statsmodel_model.params

    # Calculating the x-value at which y is zero for each coefficient
    # Excluding the intercept (usually the first coefficient)
    x_values = {}
    for variable in coefficients.index[1:]:  # Skip the intercept
        coefficient = coefficients[variable]
        intercept = coefficients[0]
        if coefficient != 0:
            x_value = -intercept / coefficient
            x_values[variable] = x_value
        else:
            x_values[variable] = None  # Undefined if coefficient is 0

    return x_values

def saddle_binarization(data_df, 
                    x_one, x_one_under_mean, x_one_over_mean, x_one_split_point,
                    x_two, x_two_under_mean, x_two_over_mean, x_two_split_point,
                    response, 
                    binarize=True, plot_error_bars=True,
                    x_label='x_label', y_label='y_label',
                    save=False, out_dir=None):
    """
    Function to create an interaction plot.
    
    Parameters
    ----------
    data_df : pandas.DataFrame
        The dataframe containing the data.
    x_one, x_two : str
        Column names of the two factors.
    x_one_under_mean, x_two_under_mean : str
        Labels to be used when the values of x_one and x_two are under the mean, respectively.
    x_one_over_mean, x_two_over_mean : str
        Labels to be used when the values of x_one and x_two are over the mean, respectively.
    x_one_split_point, x_two_split_point: int | None
        Value to split the data of x by. If None, then x will be split by mean
    response : str
        Column name of the outcome variable.
    binarize : bool, optional
        Whether to convert x_one and x_two into binary variables.
    plot_error_bars : bool, optional
        Whether to plot error bars representing SEM.
    x_label, y_label : str, optional
        Labels for the x-axis and y-axis.
    save : bool, optional
        Whether to save the plot.
    out_dir : str, optional
        Directory where the plot will be saved.

    Returns
    -------
    pandas.DataFrame
        The original dataframe with modified x_one and x_two if binarize is True.
    """
    
    # Binarize x_two variable
    if binarize:
        if x_two_split_point is not None:
            data_df[x_two] = np.where(data_df[x_two] <= x_two_split_point, f'{x_two_under_mean}', f'{x_two_over_mean}')
        else:
            data_df[x_two] = np.where(data_df[x_two] <= data_df[x_two].mean(), f'{x_two_under_mean}', f'{x_two_over_mean}')
    
    # Binarize x_one variable
    if binarize:
        if x_one_split_point is not None:
            data_df[x_one] = np.where(data_df[x_one] <= x_one_split_point, f'{x_one_under_mean}', f'{x_one_over_mean}')
        else:
            data_df[x_one] = np.where(data_df[x_one] <= data_df[x_one].mean(), f'{x_one_under_mean}', f'{x_one_over_mean}')
    
    # Map the x_two categories to numbers for plotting
    mapping = {x_two_under_mean: 0, x_two_over_mean: 1}
    data_df[x_two + '_mapped'] = data_df[x_two].map(mapping)
    
    # Extracting means and SEM for the binarized groups
    means = data_df.groupby([x_one, x_two + '_mapped'])[response].mean()
    sem = data_df.groupby([x_one, x_two + '_mapped'])[response].sem()

    # Plotting the interaction plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_dict = {f'{x_one_under_mean}': 'Blue', f'{x_one_over_mean}': 'Red'}

    for group, color in colors_dict.items():
        group_data = [(0 if combo[1] == 0 else 1, means[combo], sem[combo]) 
                      for combo in means.index if combo[0] == group]
        group_data.sort(key=lambda x: x[0])  # Sort by x-value for plotting
        
        x_vals = [item[0] for item in group_data]
        y_vals = [item[1] for item in group_data]
        y_errs = [item[2] for item in group_data] if plot_error_bars else None
        
        ax.errorbar(x_vals, y_vals, yerr=y_errs, color=color, fmt='-o', capsize=5, capthick=2, elinewidth=2, label=group, linestyle='-')

    # Setting labels and other plot properties
    ax.set_xticks([0, 1])
    ax.set_xticklabels([x_two_under_mean, x_two_over_mean])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(loc='best', frameon=False)

    # Displaying the plot
    if save:
        fig.savefig(os.path.join(out_dir, '2D_interaction_figure_pd.png'), transparent=False)
        fig.savefig(os.path.join(out_dir, '2D_interaction_figure_pd.svg'), transparent=False)
        print(f'saved to: {os.path.join(out_dir, "2D_interaction_figure.png")}')
    return plt