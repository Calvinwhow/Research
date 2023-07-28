import pandas as pd
import os

def remove_column_spaces(df):
    # Making a copy of the DataFrame
    processed_df = df.copy()
    # Replacing spaces with underscores in column names
    processed_df.columns = [col.replace(" ", "_") for col in processed_df.columns]
    return processed_df

def add_prefix_to_numeric_cols(data_df, prefix='var_'):
    """
    This function renames columns that start with a number by adding a prefix.

    Parameters:
    - data_df: DataFrame, the DataFrame to modify.
    - prefix: str, optional, the prefix to add to column names that start with a number.

    Returns:
    - DataFrame with modified column names.
    """
    new_columns = {col: prefix + col if col[0].isdigit() else col for col in data_df.columns}
    data_df = data_df.rename(columns=new_columns)
    return data_df

def replace_hyphens(data_df):
    """
    This function replaces hyphens in column names with underscores.

    Parameters:
    - data_df: DataFrame, the DataFrame to modify.

    Returns:
    - DataFrame with modified column names.
    """
    data_df.columns = [col.replace('-', '_') for col in data_df.columns]
    return data_df

def column_names_to_str(data_df):
    """
    This function replaces numbers in column names with strings.

    Parameters:
    - data_df: DataFrame, the DataFrame to modify.

    Returns:
    - DataFrame with modified column names.
    """
    data_df.columns = [str(col) for col in data_df.columns]
    return data_df

def save_dataframes_to_csv(outcome_dfs, covariate_dfs, voxelwise_dfs, path_to_dataframes):
    """
    Saves DataFrames to CSV files and returns the paths.

    Parameters:
    - outcome_dfs (list): A list of outcome DataFrames.
    - covariate_dfs (list): A list of covariate DataFrames.
    - voxelwise_dfs (list): A list of voxelwise DataFrames.
    - path_to_dataframes (str): The directory where the DataFrames will be saved.

    Returns:
    - dict: A dictionary containing lists of paths for each DataFrame type.
    """

    # Create the directory if it doesn't exist
    os.makedirs(path_to_dataframes, exist_ok=True)

    # A dictionary to store the paths
    paths = {"outcomes": [], "covariates": [], "voxelwise": []}

    # Iterate over the DataFrames, save them to CSV, and store the paths
    for i, df in enumerate(outcome_dfs):
        file_path = os.path.join(path_to_dataframes, f"outcome_data_{i+1}.csv")
        df.to_csv(file_path, index=True)
        paths["outcomes"].append(file_path)

    for i, df in enumerate(covariate_dfs):
        file_path = os.path.join(path_to_dataframes, f"covariate_data_{i+1}.csv")
        df.to_csv(file_path, index=True)
        paths["covariates"].append(file_path)

    for i, df in enumerate(voxelwise_dfs):
        file_path = os.path.join(path_to_dataframes, f"voxelwise_data_{i+1}.csv")
        df.to_csv(file_path, index=True)
        paths["voxelwise"].append(file_path)

    return paths

def preprocess_colnames_for_regression(data_df):
    data_df = column_names_to_str(data_df)
    data_df = remove_column_spaces(data_df)
    data_df = add_prefix_to_numeric_cols(data_df)
    data_df = replace_hyphens(data_df)
    return data_df