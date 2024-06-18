## Paths Input Here
from nimlab import datasets as nimds
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os 
def paths_to_input_files(path_1=None, path_2=None, analysis_name=None):
    analysis = "default_analysis"

    out_dir = os.path.join(os.path.dirname(path_1), f'{analysis}')
    if os.path.isdir(out_dir) != True:
        os.makedirs(out_dir)
    print('I will save to:', out_dir)
    return path_1, path_2, out_dir

import re
import os
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from tqdm import tqdm
from nilearn import image
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti

class GiiNiiFileImport:
    """
    A versatile class for importing and processing NIFTI and GIFTI files into NumPy arrays.

    This class is designed to handle the import of NIFTI and GIFTI files, convert them into NumPy arrays,
    and provide options for customizing the import process, handling special values, and generating column names.

    Parameters:
    -----------
    import_path : str
        The path to the directory containing the files to be imported or the path to a CSV file with file paths.
    subject_pattern : str, optional
        A regular expression pattern indicating the part of the file path to be used as the subject ID.
    process_special_values : bool, optional
        Whether to handle NaNs and infinities in the data without significantly biasing the distribution.
    file_column : str, optional
        The name of the column in the CSV file that stores file paths (required if importing from CSV).
    file_pattern : str, optional
        A file pattern to filter specific files in a folder (e.g., '*.nii.gz').

    Attributes:
    -----------
    import_path : str
        The provided import path.
    subject_pattern : str
        The subject ID pattern.
    process_special_values : bool
        Indicates whether special values should be processed.
    file_column : str
        The name of the column in the CSV file storing file paths (if applicable).
    file_pattern : str
        The file pattern for filtering files in a folder.
    matrix_df : pandas.DataFrame
        A DataFrame to store imported data.
    seen_names : set
        A set to track unique column names.
    pattern : re.Pattern
        A compiled regular expression pattern for extracting subject IDs.

    Methods:
    --------
    - generate_unique_column_name(file_path): Generates a unique column name based on the file path.
    - generate_name(file_path): Generates a name based on the file path and subject ID pattern.
    - handle_special_values(data): Handles NaNs and infinities in data without significantly biasing the distribution.
    - import_nifti_to_numpy_array(file_path): Imports a NIFTI file and converts it to a NumPy array.
    - import_gifti_to_numpy_array(file_path): Imports a GIFTI file and converts it to a NumPy array.
    - identify_file_type(file_path): Identifies whether a file is NIFTI or GIFTI based on its extension.
    - import_matrices(file_paths): Imports multiple files and stores them in the DataFrame.
    - import_from_csv(): Imports data from a CSV file using the specified file column.
    - import_from_folder(): Imports data from files in a folder based on the provided file pattern.
    - detect_input_type(): Detects whether the input is a CSV file or a folder.
    - import_data_based_on_type(): Imports data based on the detected input type.
    - run(): Orchestrates the import process based on the input type and returns the DataFrame.

    Note:
    -----
    This class provides a flexible way to import and process neuroimaging data in NIFTI and GIFTI formats,
    making it suitable for various data analysis tasks.

    """
    def __init__(self, import_path, subject_pattern='', process_special_values=True, file_column: str=None, file_pattern: str=None):
        self.import_path = import_path
        self.file_pattern = file_pattern
        self.file_column = file_column
        self.subject_pattern = subject_pattern
        self.process_special_values = process_special_values
        self.matrix_df = pd.DataFrame({})
        self.seen_names = set()
        self.pattern = re.compile(f"{self.subject_pattern}(\d+)")
    
    def generate_unique_column_name(self, file_path: str) -> str:
        base_name = os.path.basename(file_path)
        if base_name in self.seen_names:
            name = os.path.join(os.path.basename(os.path.dirname(file_path)), base_name)
        else:
            name = base_name
            self.seen_names.add(base_name)
        return name
    
    def match_id(self, file_path):
        match = self.pattern.search(file_path)
        if match:
            subject_id = match.group()
            return subject_id + '_' + os.path.basename(file_path)
        
    def generate_name(self, file_path: str):
        """
        Generates a name based on the file path and an optional subject ID pattern.

        Parameters:
        -----------
        file_path : str
            The file path from which to extract the base name or subject ID.
        subject_id_pattern : str, optional
            A pattern indicating the part of the file_path to be used as the subject ID.

        Returns:
        --------
        str
            A generated name based on the file path and the specified pattern.
        """

        if self.subject_pattern != '':
            name = self.match_id()
        else:
            name = self.generate_unique_column_name(file_path)
        return name 
    
    def handle_special_values(self, data):
        """
        Handles NaNs and infinities in the data without significantly biasing the distribution.

        Args:
            data: NumPy array with the data.

        Returns:
            Updated data with NaNs and infinities handled.
        # """
        max_val = np.nanmax(data)
        min_val = np.nanmin(data)
        data = np.nan_to_num(data, nan=0.0, posinf=max_val, neginf=min_val)
        return data
    
    def import_nifti_to_numpy_array(self, file_path):
        '''
        Does what it says. Just provide the absolute filepath.
        Args:
            filepath: absolute path to the file to import
        Returns:
            nifti_data: nifti_data as a numpy array
        '''
        try:
            # Load the NIfTI image using nilearn
            nifti_img = image.load_img(file_path)

            # Convert the NIfTI image to a NumPy array
            nifti_data = nifti_img.get_fdata().flatten()

            # Return the NIfTI image
            return nifti_data
        except Exception as e:
            print("Error:", e)
            return None

    def import_gifti_to_numpy_array(self, file_path):
        """
        Imports a GIFTI file and converts it to a NumPy array.

        Args:
            filepath: Absolute path to the GIFTI file.

        Returns:
            gifti_data: Data from the GIFTI file as a NumPy array.
        """
        try:
            # Load the GIFTI file
            gifti_img = nib.load(file_path)

            # Extract the data array from the GIFTI image
            # This assumes the data is in the first darray; adjust as needed for your files
            gifti_data = gifti_img.darrays[0].data.flatten()

            return gifti_data
        except Exception as e:
            print("Error:", e)
            return None
        
    def identify_file_type(self, file_path: str) -> str:
        """
        Identifies whether a file is NIFTI or GIFTI based on its extension.

        Parameters:
        -----------
        file_path : str
            The file path to be checked.

        Returns:
        --------
        str
            'nii' if the file is a NIFTI file, 'gii' if it's a GIFTI file, or an empty string if neither.
        """
        if file_path.lower().endswith('.nii') or file_path.lower().endswith('.nii.gz'):
            return 'nii'
        elif file_path.lower().endswith('.gii') or file_path.lower().endswith('.gii.gz'):
            return 'gii'
        else:
            return ''

    def import_matrices(self, file_paths):
        for file_path in file_paths:
            # Load and Check if File Path Exists
            path = self.identify_file_type(file_path)
            if path == 'nii':
                data = self.import_nifti_to_numpy_array(file_path)
            elif path == 'gii':
                data = self.import_gifti_to_numpy_array(file_path)
            else:
                raise ValueError("Invalid Path Type.")
            
            # Convert NaNs and Infs if specified
            if self.process_special_values:
                data = self.handle_special_values(data)
                
            # Generate unique column name
            name = self.generate_name(file_path)
            # Add data to DataFrame
            self.matrix_df[name] = data 
        return self.matrix_df

    def import_from_csv(self):
        print(f'Attempting to import from: {os.path.basename(self.import_path)}')
        if self.import_path is None:
            raise ValueError ("Argument file_column is None. Please specify file_column='column_storing_file_paths.")
        self.paths = pd.read_csv(self.import_path)[self.file_column].tolist()
        return self.import_matrices(self.paths)

    def import_from_folder(self):
        print(f'Attempting to import from: {self.import_path}/{self.file_pattern}')
        if self.file_pattern == '':
            raise ValueError ("Argument file_pattern is empty. Please specify file_pattern='*my_file*patter.nii.gz'")
        glob_path = os.path.join(self.import_path, self.file_pattern)
        file_paths = glob(glob_path)
        self.file_paths = file_paths
        return self.import_matrices(file_paths)

    def detect_input_type(self):
        """
        Detects whether the input_path is a CSV file or a folder.

        Parameters:
        -----------
        input_path : str
            The input path to be checked.

        Returns:
        --------
        str
            'csv' if the input is a CSV file, 'folder' if it's a folder, or 'unsupported' if neither.
        """
        if self.import_path.lower().endswith('.csv'):
            self.import_type = 'csv'
        else:
            self.import_type = 'folder'
    
    def import_data_based_on_type(self):
        self.detect_input_type()
        if self.import_type == 'csv':
            # Input is a CSV file
            return self.import_from_csv()
        elif self.import_type == 'folder':
            # Input is a folder
            return self.import_from_folder()
        else:
            raise ValueError("Invalid input type")
        
    @staticmethod
    def save_files(dataframe, file_paths, dry_run=True, file_suffix=None):
        """
        Convenience saving function. Allows saving files after acting upon them. 
        """
        for i, file_path in tqdm(enumerate(file_paths), desc='Saving files'):
            out_dir = os.path.dirname(file_path)
            nifti_name = os.path.splitext(os.path.basename(file_path))[0] + (file_suffix if file_suffix is not None else '')
            if dry_run:
                print(f"Saving to: {os.path.join(out_dir, nifti_name)}")
            else:
                view_and_save_nifti(dataframe.iloc[:, i], out_dir=out_dir, output_name=nifti_name, silent=True)
    
    @staticmethod
    def mask_dataframe(df: pd.DataFrame, mask_path: str=None, threshold: float=0):
        """
        Simple masking function.
        """
        if mask_path is None:
            mask = nimds.get_img("mni_icbm152")
        else:
            mask = nib.load(mask_path)
            
        mask = mask.get_fdata().flatten()
        mask_indices = mask > threshold
        masked_df = df.loc[mask_indices, :]

        return mask, mask_indices, masked_df
    
    @staticmethod
    def unmask_dataframe(df: pd.DataFrame, mask_path: str=None, threshold: float=0):
        """
        Simple unmasking function.
        """
        if mask_path is None:
            mask = nimds.get_img("mni_icbm152")
        else:
            mask = nib.load(mask_path)
        mask = mask.get_fdata().flatten()
        mask_indices = mask > threshold
        
        unmasked_df = pd.DataFrame(index=mask, columns=df.columns, data=0)
        unmasked_df.iloc[mask_indices, :] = df
        return unmasked_df
    
    @staticmethod
    def splice_colnames(df, pre, post):
        raw_names = df.columns
        name_mapping = {}
        # For each column name in the dataframe
        for name in raw_names:
            new_name = name  # Default to the original name in case it doesn't match any split command
            new_name = new_name.split(pre)[1]
            new_name = new_name.split(post)[0]
            
            # Add the original and new name to the mapping
            name_mapping[name] = new_name
        return df.rename(columns=name_mapping)
    
    
    def run(self):
        self.import_data_based_on_type()
        return self.matrix_df
