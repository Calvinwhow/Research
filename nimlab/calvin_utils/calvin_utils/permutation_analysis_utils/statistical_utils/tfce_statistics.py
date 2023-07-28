'''
TFCalculator
---------------
This module provides the TFCalculator class for performing Threshold-Free Cluster Enhancement (TFCE) on 3D Nifti images.
This can also accept a 4D nifti, where the 4th dimension represents permutations. 

Example Usage:

    from TFCalculator import TFCalculator

    calculator = TFCalculator(E=0.5, H=2, dh=0.1) # These are default parameters
    calculator.process_folder('/path/to/nii/files')

Notes:
- The TFCalculator class uses the standard formula for TFCE and applies it to each voxel in the 3D image. The image data should represent a 3D map of test statistics (such as T-statistics).
- In this implementation, the E and H parameters are customizable, as well as the step size for the height increment (dh).
- The class assumes the input files are .nii format and outputs the results as new .nii files with '_tfce' appended to the original filenames.
'''

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from joblib import Parallel, delayed

class TFCalculator:
    """
    This class is responsible for calculating Threshold-Free Cluster Enhancement (TFCE) for Nifti files.
    
    Reference: 
    TFCE Neuroimage paper: 
    White Paper: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.fmrib.ox.ac.uk/datasets/techrep/tr08ss1/tr08ss1.pdf 
    Cython comparison: https://github.com/trislett/TFCE_mediation/blob/master/tfce_mediation/tmanalysis/STEP_1_voxel_tfce_multiple_regression.py 
    MatLab comparison: https://github.com/markallenthornton/MatlabTFCE
    
    To Enable Inference Upon Voxels Within the Clusters, Implement This: 
        All-Resoutions Inference for Brain Imaging: https://www.sciencedirect.com/science/article/abs/pii/S105381191830675X?via%3Dihub
    """
    def __init__(self, E=0.5, H=2, dh=0.1):
        """
        Initializes a new instance of the TFCalculator class.
        
        :param E: The E parameter of the TFCE formula. Default is 0.5.
        :param H: The H parameter of the TFCE formula. Default is 2.
        :param dh: The step size for the height increment in the TFCE formula. Default is 0.1.
        """
        self.E = E
        self.H = H
        self.dh = dh
        
    def tfce_transform_indexed(self, stats):
        """
        Performs the TFCE transformation on a given map of abitrary statistics.

        :param tstats: A 3D numpy array containing the arbitrary statistics.
        :return: A 3D numpy array containing the TFCE values.
        """
        tfce_map = np.zeros_like(stats)
        cluster_labels, num_clusters = label(stats > 0)

        for cluster_id in range(1, num_clusters + 1):
            cluster_map = cluster_labels == cluster_id
            max_height = stats[cluster_map].max()
            height_steps = np.arange(self.dh, max_height + self.dh, self.dh)

            # Get all clusters at each height in one go
            clusters_at_heights = (stats[..., None] >= height_steps) & cluster_map[..., None]

            # Calculate extents for all clusters at all heights in one go
            extents = clusters_at_heights.sum(axis=(0, 1, 2))

            # Add contributions from all heights for this cluster in one go
            tfce_map[cluster_map] += np.sum((extents ** self.E * height_steps ** self.H * self.dh) * clusters_at_heights, axis=-1)

        return tfce_map

    def tfce_transform_looped(self, tstats):
        """
        Performs the TFCE transformation on a given map of T-statistics.
        
        :param tstats: A 3D numpy array containing the T-statistics.
        :return: A 3D numpy array containing the TFCE values.
        """
        tfce_map = np.zeros_like(tstats)
        cluster_labels, num_clusters = label(tstats > 0)

        for cluster_id in range(1, num_clusters+1):
            cluster_map = cluster_labels == cluster_id
            max_height = tstats[cluster_map].max()
            height_steps = np.arange(self.dh, max_height + self.dh, self.dh)

            for h in height_steps:
                cluster_at_height = cluster_map & (tstats >= h)
                extent = cluster_at_height.sum()
                tfce_map[cluster_at_height] += extent ** self.E * h ** self.H * self.dh

        return tfce_map

    def compute_tfce(self, nii_file, no_save=False):
        """
        Loads a .nii file, applies the TFCE transform, and optionally saves the result to a new .nii file.
        
        This method loads a .nii file, computes the TFCE of the 3D image in the file, and either
        saves the TFCE-transformed image to a new .nii file or returns the maximum TFCE value 
        without saving the file.
        
        :param nii_file: The path to the .nii file.
        :param no_save: A boolean indicating whether to save the TFCE-transformed image. If True,
                        the method returns the maximum TFCE value and does not save the image. 
                        If False, the method saves the TFCE-transformed image and does not return 
                        the maximum TFCE value. Default is False.
        :return: If no_save is True, returns the maximum TFCE value. Otherwise, does not return a value.
        """
        try:
            img = nib.load(nii_file)
            data = img.get_fdata()
            tfce_data = self.tfce_transform_indexed(data)  # Or use self.tfce_transform_looped(data) as per your needs
            tfce_img = nib.Nifti1Image(tfce_data, img.affine)
            tfce_file = os.path.splitext(nii_file)[0] + '_tfce.nii.gz'
            if no_save:
                return np.max(tfce_data)
            else:
                nib.save(tfce_img, tfce_file)
        except Exception as e:
            print(f"Failed to process file {nii_file}. Error: {str(e)}")

    def load_nii(self, nii_file):
            """
            Loads a .nii file and returns the corresponding numpy array.
            
            :param nii_file: The path to the .nii file.
            :return: A numpy array containing the image data.
            """
            try:
                img = nib.load(nii_file)
                return img.get_fdata()
            except Exception as e:
                print(f"Failed to load file {nii_file}. Error: {str(e)}")
                return None

    def load_and_process(self, nii_file):
        '''
        Takes nifti file path, loads it and processes it.
        
        :param nii_file: The path to the .nii file.
        :return: A numpy array containing the image data.
        '''
        data = self.load_nii(nii_file)
        self.compute_tfce(data)

        
    def process_folder(self, folder):
        """
        Processes all .nii files in a given folder.
        
        :param folder: The path to the folder containing the .nii files.
        """
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            return

        nii_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.nii')]

        if not nii_files:
            print(f"No .nii files found in the folder {folder}.")
            return

        Parallel(n_jobs=-1)(delayed(self.load_and_process)(nii_file) for nii_file in nii_files)

    def extract_max_values(self, file_path):
        """
        Extracts the maximum Threshold-Free Cluster Enhancement (TFCE) values from each 3D image 
        in a 4D statistical image and returns them in an array.
        
        For each 3D image in the 4D statistical image, this method computes the TFCE and extracts 
        the maximum TFCE value. It does not save the TFCE-transformed images.
        
        :argument: filepath: Path to the file containing the 4D statistical image
        
        :return: A 1D numpy array containing the maximum TFCE value from each 3D image.
        """
        data = self.load_nii(file_path)
        if data is not None:
            max_values = np.full(self.data.shape[-1], np.nan)
            for i in range(self.data.shape[-1]):
                max_values[i] = self.compute_tfce(self.data[..., i], no_save=True)
            return max_values
        else:
            print(f"Failed to progress. Nifti generated array=None.")
            return None
