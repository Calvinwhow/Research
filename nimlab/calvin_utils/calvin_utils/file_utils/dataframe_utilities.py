import os
import pandas as pd
from natsort import natsorted

def natsort_df(df):
    #Sort the Dataframe            
    df = df.reindex(index=natsorted(df.index))
    sorted_df = df.reindex(columns=natsorted(df.columns))
    return sorted_df

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

def extract_and_rename_subject_id(dataframe, split_command_dict):
    """
    Renames the columns of a dataframe based on specified split commands.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe whose columns need to be renamed.
    - split_command_dict (dict): A dictionary where the key is the split string 
                                 and the value is the order to take after splitting 
                                 (0 for before the split, 1 for after the split, etc.).

    Returns:
    - pd.DataFrame: Dataframe with renamed columns.

    Example:
    >>> data = {'subject_001': [1, 2, 3], 'patient_002': [4, 5, 6], 'control_003': [7, 8, 9]}
    >>> df = pd.DataFrame(data)
    >>> split_commands = {'_': 1}
    >>> new_df = extract_and_rename_subject_id(df, split_commands)
    >>> print(new_df.columns)
    Index(['001', '002', '003'], dtype='object')
    """

    raw_names = dataframe.columns
    name_mapping = {}

    # For each column name in the dataframe
    for name in raw_names:
        new_name = name  # Default to the original name in case it doesn't match any split command

        # Check each split command to see if it applies to this column name
        for k, v in split_command_dict.items():
            if k in new_name:
                new_name = new_name.split(k)[v]

        # Add the original and new name to the mapping
        name_mapping[name] = new_name

    # Rename columns in the dataframe based on the mapping
    return dataframe.rename(columns=name_mapping)
