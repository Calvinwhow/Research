import numpy as np
from glob import glob
from nilearn import image
import os
import pandas as pd

import pandas as pd
import numpy as np
import nibabel as nib

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
    '''
    Reads a CSV file containing paths to nifti files, imports the nifti files, flattens them,
    removes NaNs, and creates a dataframe in the specified format.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing paths to nifti files.
        
    Returns:
    --------
    pd.DataFrame
        A dataframe where columns represent flattened nifti files and rows represent voxels.
        All values are zero, except for lesions which are binarized at 1.
    
    '''
    # Read the CSV file
    file_paths = pd.read_csv(csv_path)
    
    # Initialize a DataFrame
    matrix_df = pd.DataFrame({})

    # Initialize an empty list to store flattened nifti data
    names = []
    
    # Iterate through the file paths and import nifti files
    for index, row in file_paths.iterrows():
        file_path = row.values[0]  # Assuming the file paths are in the first column of the CSV

        # Ensure the file exists before trying to open it
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        print(file_path)
        # print('I found : ', file)
        img = image.load_img(file_path)
        #Organize files into manipulatable dataframes
        data = img.get_fdata()
        
        name = os.path.basename(file_path).split('_tome')[0]
        try:
            name = name.split('sub-')[1]
            name = name.split('uvat')[0]
        except:
            pass
        try:
            name = name.split('_vat')[0]
        except:
            print('Could not split file name')
        matrix_df[name] = data.flatten()
        names.append(name)
    return matrix_df

def import_matrices_from_folder(connectivity_path, file_pattern='', convert_nan_to_num=None):
    glob_path  = os.path.join(connectivity_path, file_pattern)
    print('I will search: ', glob_path)

    globbed = glob(glob_path)
    #Identify files of interest
    matrix_df = pd.DataFrame({})
    names = []
    for file in globbed:
        # print('I found : ', file)
        img = image.load_img(file)
        #Organize files into manipulatable dataframes
        data = img.get_fdata()
        if convert_nan_to_num is not None:
            data = np.nan_to_num(data, nan=convert_nan_to_num['nan'], posinf=convert_nan_to_num['posinf'], neginf=convert_nan_to_num['neginf'])
        else:
            pass
        name = os.path.basename(file)
        matrix_df[name] = data.flatten()
        names.append(name)
    return matrix_df
