import numpy as np
import nibabel as nib
import pandas as pd
"""
This is a piece of software to be used with a design matrix composed of the paths to 4-dimensional nifti files.
It will perform a simple linear regression and return the t-statistic of the given regressor (the given 4D nifti).
"""

class VoxelwiseRegression:
    """
    Perform voxelwise regression on 4D NIfTI data.
    
    Attributes:
        csv_path (str): Path to the CSV file containing paths to 4D NIfTI files for each variable.
        mask_path (str, optional): Path to the 3D NIfTI file containing the mask.
        design_matrix_paths (DataFrame): DataFrame containing paths to 4D NIfTI files for each variable.
        nifti_data (dict): Dictionary containing the loaded 4D NIfTI data for each variable.
    
    Methods:
        import_nifti(): Load 4D NIfTI data for each variable.
        apply_mask(): Apply the 3D mask to each 4D NIfTI data.
        preprocess(): Placeholder for preprocessing steps.
        run_regression(): Perform the voxelwise regression.
        run(): Main method to run the full pipeline.
    """
    def __init__(self, csv_path, mask_path=None):
        self.csv_path = csv_path
        self.mask_path = mask_path
        self.design_matrix_paths = pd.read_csv(csv_path)
        self.nifti_data = {}
        
    def import_nifti(self):
        """Load 4D NIfTI data for each variable."""
        for var_name, file_path in self.design_matrix_paths.iterrows():
            self.nifti_data[var_name] = nib.load(file_path[0]).get_fdata()
    
    def apply_mask(self):
        """Apply the 3D mask to each 4D NIfTI data."""
        if self.mask_path:
            mask = nib.load(self.mask_path).get_fdata()
            mask_indices = np.nonzero(mask)
            
            # Use broadcasting to apply the mask to all 4D arrays
            for var_name in self.nifti_data:
                self.nifti_data[var_name] = self.nifti_data[var_name][..., mask_indices]
    
    def preprocess(self):
        """Placeholder for preprocessing steps."""
        pass
    
    def run_regression(self):
        """Perform the voxelwise regression."""
        # Rest of the code remains the same

    def run(self):
        """Main method to run the full pipeline."""
        self.import_nifti()
        self.apply_mask()
        self.preprocess()
        coeffs_4D = self.run_regression()
        return coeffs_4D
