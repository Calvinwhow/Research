import numpy as np
from glob import glob
from nilearn import image
import os
import pandas as pd
from calvin_utils.nifti_utils.matrix_utilities import import_nifti_to_numpy_array

import nibabel as nib

def generate_unique_column_name(file_path: str, seen_names: set) -> str:
    """
    Generates a unique column name based on the file path and a set of already seen names.
    
    Parameters:
    -----------
    file_path : str
        The file path from which to extract the base name.
    seen_names : set
        A set containing names that have already been seen.
        
    Returns:
    --------
    str
        A unique column name.
    """
    base_name = os.path.basename(file_path)
    if base_name in seen_names:
        name = os.path.join(os.path.basename(os.path.dirname(file_path)), base_name)
    else:
        name = base_name
        seen_names.add(base_name)
    return name

def import_matrices_from_df_series(df_series):
    from nilearn import image
    # Iterate through the file paths and import nifti files
    matrix_df = pd.DataFrame({})
    for index, row in df_series.iterrows():
        file_path = row.values[0]  # Assuming the file paths are in the first column of the CSV

        # Ensure the file exists before trying to open it
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        img = image.load_img(file_path)
        #Organize files into manipulatable dataframes
        data = img.get_fdata()
        
        name = index
        matrix_df[name] = data.flatten()
    return matrix_df

def import_matrices_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads a CSV file containing paths to NIFTI files, imports the NIFTI files, flattens them,
    and creates a DataFrame in the specified format.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing paths to NIFTI files.
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame where columns represent flattened NIFTI files and rows represent voxels.
    """
    # Read the CSV file
    file_paths = pd.read_csv(csv_path)
    
    # Initialize a DataFrame
    matrix_df = pd.DataFrame({})
    
    # Initialize a set to store seen base names
    seen_names = set()
    
    # Iterate through the file paths and import NIFTI files
    for index, row in file_paths.iterrows():
        file_path = row.values[0]  # Assuming the file paths are in the first column of the CSV

        # Load and Check if File Path Exists
        data = import_nifti_to_numpy_array(file_path)
        
        # Generate unique column name
        name = generate_unique_column_name(file_path, seen_names)
        
        # Add data to DataFrame
        matrix_df[name] = data.flatten()
        
    return matrix_df

def get_subject_id_from_path(file_path, subject_id_index):
    """
    Extracts the subject ID from a file path based on a specified index.

    Parameters:
    - file_path (str): The full path of the file.
    - subject_id_index (int): The index of the part of the file path that represents the subject ID.

    Returns:
    - str: The extracted subject ID, or an empty string if the index is out of range.
    """

    # Split the path into parts
    path_parts = file_path.split(os.sep)

    # Check if the specified index is within the range of path_parts
    if 0 <= subject_id_index < len(path_parts):
        return path_parts[subject_id_index]
    else:
        return ""  # Return an empty string if the index is out of range


def import_matrices_from_folder(connectivity_path, file_pattern='', convert_nan_to_num=None, subject_id_index=None):
    glob_path  = os.path.join(connectivity_path, file_pattern)
    print('I will search: ', glob_path)

    globbed = glob(glob_path)
    #Identify files of interest
    matrix_df = pd.DataFrame({})
    names = []
    seen_names = set()
    for file in globbed:
        # print('I found : ', file)
        img = image.load_img(file)
        #Organize files into manipulatable dataframes
        data = img.get_fdata()
        if convert_nan_to_num is not None:
            data = np.nan_to_num(data, nan=convert_nan_to_num['nan'], posinf=convert_nan_to_num['posinf'], neginf=convert_nan_to_num['neginf'])
        else:
            pass
        if subject_id_index is not None:
            id = get_subject_id_from_path(file, subject_id_index=subject_id_index)
            name = id + '_' + os.path.basename(file)
        else:
            name = generate_unique_column_name(file, seen_names)
        matrix_df[name] = data.flatten()
        names.append(name)
    return matrix_df


