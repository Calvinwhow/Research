import numpy as np
import pandas as pd
from nimlab import datasets as nimds
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti
from calvin_utils.nifti_utils.matrix_utilities import threshold_matrix, import_nifti_to_numpy_array, unmask_matrix, apply_mask_to_dataframe

class ConjunctionMap:
    def __init__(self, nift_path_1, nift_path_2, threshold1, threshold2, direction1, direction2, mask_path=None, 
                 output_dir='generated_nifti', output_name='conjunction_map.nii'):
        """
        Initialize the class with two data frames, two threshold values, and two directions for thresholding.

        Parameters:
        nift_path_1, nift_path_2 : path
            Path to NIFTI files and rows represent voxels.
        threshold1, threshold2 : float
            Threshold values for the data frames.
        direction1, direction2 : str
            Directions for thresholding ('keep_above', 'keep_below', etc.).
        mask_path : str, optional
            Shared path to the NIFTI mask file.
        output_dir : str
            The path to save the output NIfTI file.
        output_name : str, optional
            The name to use when saving the NIfTI file.
        """
        self.df1 = import_nifti_to_numpy_array(nift_path_1).flatten()
        self.df2 = import_nifti_to_numpy_array(nift_path_2).flatten()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.direction1 = direction1
        self.direction2 = direction2
        self.mask_path = mask_path
        self.output_dir = output_dir
        self.output_name = output_name
        
        if mask_path:
            self.mask = import_nifti_to_numpy_array(mask_path)  # Replace with your import function
        else:
            mask_img = nimds.get_img("mni_icbm152")
            self.mask = mask_img.get_fdata()
            
    def threshold_matrices(self):
        """
        Apply threshold to both data frames using the imported function threshold_matrix.
        """
        self.df1_thresholded = threshold_matrix(self.df1, self.threshold1, direction=self.direction1)
        self.df2_thresholded = threshold_matrix(self.df2, self.threshold2, direction=self.direction2)
        
    def perform_conjunction(self):
        """
        Perform conjunction analysis to identify strictly overlapping voxels.
        """
        self.df1_thresholded = self.df1_thresholded.astype(bool)
        self.df2_thresholded = self.df2_thresholded.astype(bool)
        
        self.conjunction_df = self.df1_thresholded & self.df2_thresholded
        print(self.conjunction_df.shape)

    def unmask_matrix(self):
        """
        Unmask the DataFrame to insert values back into their original locations in a full brain mask.
        """
        if not isinstance(self.conjunction_df, pd.DataFrame):
            self.conjunction_df = pd.DataFrame(self.conjunction_df)
        self.unmasked_df = unmask_matrix(apply_mask_to_dataframe(self.conjunction_df, mask_path=self.mask_path), self.mask_path)
    
    def generate_save_and_view_nifti(self, output_dir=None, output_name=None):
        """
        Generate, save, and view the NIFTI file.
        """
        self.img = view_and_save_nifti(self.unmasked_df, out_dir=output_dir, output_name=output_name)
        
    def run(self):
        """
        Run all methods in sequence.
        """
        self.threshold_matrices()
        self.perform_conjunction()
        self.unmask_matrix()
        self.generate_save_and_view_nifti(output_dir=self.output_dir, output_name=self.output_name)
        
        return self.img
