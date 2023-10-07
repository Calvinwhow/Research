import numpy as np
import pandas as pd
from nimlab import datasets as nimds
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti
from calvin_utils.nifti_utils.matrix_utilities import view_nifti_html, import_nifti_to_numpy_array, unmask_matrix

class SensitivityTestMap:
    """
    Class to process NIFTI data stored in a DataFrame.
    
    Attributes:
    -----------
    df : pd.DataFrame
        DataFrame where columns represent flattened NIFTI files and rows represent voxels.
    overlapping_studies : pd.Series
        Summation of binary voxels across studies to find overlap.
    mask : np.ndarray
        NIFTI mask to reshape the flattened data back to original shape.
        
    # # Usage
    # # Assuming df is your DataFrame
    # processor = NiftiProcessor(df)
    # processor.threshold_dataframe(0.5)
    # html_viewer = processor.view_nifti_html()
    """
    
    def __init__(self, df: pd.DataFrame, sensitivity_threshold: float, output_dir: str='generated_nifti', output_name=None, mask_path: str = None, absval=False):
        """
        Initialize with a DataFrame containing flattened NIFTI files and an optional mask.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame where columns represent flattened NIFTI files and rows represent voxels.
        mask_path : str, optional
            Path to the NIFTI mask file.
        default_mask : np.ndarray, optional
            Default mask to use if mask_path is None.
        absval: bool, optional
            Sets the input data to their absolute values.
        - sensitivity_threshold (float): 
            Value to threshold the DataFrame.
        - output_dir (str): 
            The path to save the output NIfTI file.
        - output_name (str, optional): 
            The name to use when saving the NIfTI file.
        """
        self.input_df = df
        self.sensitivity_threshold = sensitivity_threshold
        self.output_dir = output_dir
        self.output_name = output_name
        if absval:
            self.input_df = self.input_df.abs()
        if mask_path:
            self.mask = import_nifti_to_numpy_array(mask_path)  # Replace with your import function
        else:
            mask_img = nimds.get_img("mni_icbm152")
            self.mask = mask_img.get_fdata()
    
    def threshold_dataframe(self, threshold: float):
        """
        Threshold the DataFrame and calculate the number of overlapping studies at each voxel.
        
        Parameters:
        -----------
        threshold : float
            Value to threshold the DataFrame.
        """
        self.df = (self.input_df >= threshold).astype(int)
        self.overlapping_studies = self.df.sum(axis=1)
        
    def unmask_matrix(self, mask_path=None, mask_threshold=0.2, unmask_by='rows', dataframe_to_unmask_by=None):
        """
        Unmask the dataframe by inserting values back into their original locations in a full brain mask.
        """

        # Import and use the unmask_matrix function
        unmasked_df = unmask_matrix(self.overlapping_studies, mask_path, mask_threshold, unmask_by, dataframe_to_unmask_by)
        
        # Optionally, you can store the unmasked dataframe as a class attribute
        self.unmasked_df = unmasked_df
    
    def generate_save_and_view_nifti(self, output_dir=None, output_name=None):
        """
        Apply the `nifti_from_matrix` function to convert the unmasked dataframe to a NIfTI image.

        Parameters:
        - output_file (str): The path to save the output NIfTI file.
        - output_name (str, optional): The name to use when saving the NIfTI file.

        Returns:
        - pd.DataFrame: The unmasked dataframe.
        """

        # Call the nifti_from_matrix function
        self.img = view_and_save_nifti(self.unmasked_df, out_dir=output_dir, output_name=output_name)
        
    def get_max_overlap_info(self):
        """
        Get information about the rows with the maximum overlap and their contributing files.

        Returns:
        - list: List of rows (voxels) with the highest overlap.
        - list of lists: List of files (columns) that contributed to the highest overlap for each max overlap row.
        """
        if self.overlapping_studies.empty:
            raise ValueError("overlapping_studies is empty. Run threshold_dataframe first.")
        
        # Find the row(s) with the maximum value of overlapping_studies
        max_overlap_value = self.overlapping_studies.max()
        max_overlap_rows = self.overlapping_studies[self.overlapping_studies == max_overlap_value].index.tolist()

        # Check for empty or invalid index
        if not max_overlap_rows:
            print('No maximum overlap found.')
            return [], []
        
        # Initialize a list to store the contributing files for each max overlap row
        max_overlap_contributing_files = []
        
        for row_idx in max_overlap_rows:
            try:
                contributing_files = self.input_df.columns[(self.input_df.loc[row_idx] >= self.sensitivity_threshold)].tolist()
                max_overlap_contributing_files.append(contributing_files)
            except:
                max_overlap_contributing_files.append([])  # or some indication that the index was invalid

        return max_overlap_rows, max_overlap_contributing_files

    
    def run(self):
        """
        Run the entire workflow, including thresholding, unmasking, and generating a NIfTI image.

        Returns:
        - pd.DataFrame: The unmasked dataframe.
        - object: HTML viewer object for the overlapping studies.
        """
        # Threshold the DataFrame
        self.threshold_dataframe(self.sensitivity_threshold)

        # Unmask the DataFrame
        self.unmask_matrix()

        # Generate, save, and view the NIfTI image
        self.generate_save_and_view_nifti(output_dir=self.output_dir, output_name=self.output_name)

        # Find the niftis which contributed to the overlap
        max_overlap_rows, max_overlap_contributing_files = self.get_max_overlap_info()
        
        return self.unmasked_df, self.img, max_overlap_contributing_files