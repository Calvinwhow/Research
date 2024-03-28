import numpy as np
from nilearn import image
from nimlab import datasets as nimds
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from glob import glob
from calvin_utils.file_utils.import_matrices import import_matrices_from_folder
from nimlab import datasets as nimds
from nilearn import image, plotting
import nibabel as nib
import os
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import image
import re

def nifti_from_matrix(matrix, output_file, ref_file=None, use_reference=True, reference='MNI', use_affine=False, affine='MNI', output_name=None, silent=False):
    """Converts a flattened matrix to a NIfTI file using the given affine matrix.

    Args:
        matrix (ndarray): A flattened 3D or 4D matrix.
        output_file (str): The path to save the output NIfTI file.
        use_reference (bool, optional): Whether you want to use a reference image such as an MNI mask
        reference (str, optional): The reference image to use, such as the MNI152 mask
        use_affine (bool, optional): Whether to use affine transformation
        affine (ndarray, optional): A 4x4 affine matrix describing the voxel-to-space mapping. 
            Defaults to None, in which case an identity matrix will be used.
    """
    matrix = np.array(matrix)
    if use_affine:
        # Create an identity affine matrix if none is provided
        if affine is None:
            affine = np.eye(4)
        elif affine == 'MNI':
            affine = [[ -1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   9.60000000e+01],
                        [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,  -1.26000000e+02],
                        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,  -7.20000000e+01],
                        [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]
            matrix = np.reshape(matrix, (91, 109, 91), order='F')
          # Create a new image from the array and the affine matrix
        img = image.new_img_like(output_file, matrix, affine)
    elif use_reference:
        if reference == 'MNI':
            mask = nimds.get_img("mni_icbm152")
            ref_img = mask.get_fdata()
            matrix = matrix.reshape(ref_img.shape)
        else:
            ref_img = image.load_img(ref_file)
            ref_img_data = ref_img.get_fdata()
            mask = np.where(ref_img > 0, 1, 0) #Quick-and-dirty mask-generation assuming T1 image. 
            matrix = matrix.reshape(ref_img_data.shape)
        # Create a new image from the array and the affine matrix
        img = image.new_img_like(ref_niimg=mask, data=matrix)

    #Make directories
    if os.path.exists(output_file):
        pass
    else:
        os.makedirs(output_file)
    
    # Save the image to the output file
    if output_name is not None:
        img.to_filename(os.path.join(output_file, f'{output_name}.nii'))
    else:
        img.to_filename(os.path.join(output_file, 'generated_nifti.nii'))
    if silent==False:
        print('Image saved to: \n', output_file)
    return img
    
def generate_concentric_spherical_roi(subject, x, y, z, out_dir, max_radius=12):
    from nilearn import image
    from nltools.mask import create_sphere
    import numpy as np

    # Create solid sphere with max_radius
    sphere_roi = create_sphere([x, y, z], radius=max_radius)
    sphere_data = sphere_roi.get_fdata()

    # Convert the center of the sphere from MNI space to array index space
    center_idx = np.round(image.coord_transform(x, y, z, np.linalg.inv(sphere_roi.affine))).astype(int)

    # Calculate distances for each voxel in the sphere to the center
    distances = np.sqrt(np.sum((np.indices(sphere_data.shape) - center_idx.reshape(-1, 1, 1, 1))**2, axis=0))

    # Normalize distances to [0, 1] range and invert it
    distances = 1 - distances / max_radius

    # Apply mask of the sphere
    sphere_data = np.where(sphere_data!=0, distances, 0)

    # Save as a nifti image
    out_img = image.new_img_like(sphere_roi, sphere_data)

    # Save
    save_dir = os.path.join(out_dir, f'{subject}_eccentric_sphere.nii.gz')
    print('saved to: ', save_dir)
    out_img.to_filename(save_dir)

    return save_dir



def generate_eccentric_spherical_roi(subject, x, y, z, out_dir, radius=5, eccentricity=(1,1,1)):
    # Create a sphere with the given coordinates and radius
    from nltools.mask import create_sphere
    sphere_mask = create_sphere([x, y, z], radius=radius)

    # Alter the shape of the sphere to create an eccentric sphere
    data = sphere_mask.get_fdata()
    dims = data.shape

    new_dims = [int(dim * ecc) for dim, ecc in zip(dims, eccentricity)]
    new_data = np.zeros(new_dims)

    scale_factors = [new_dim / dim for new_dim, dim in zip(new_dims, dims)]
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if data[i, j, k] > 0:
                    new_i = int(i * scale_factors[0])
                    new_j = int(j * scale_factors[1])
                    new_k = int(k * scale_factors[2])
                    new_data[new_i, new_j, new_k] = data[i, j, k]
    
    eccentric_sphere = nib.Nifti1Image(new_data, sphere_mask.affine)

    # Resample the eccentric sphere to the original sphere's dimensions
    resampled_eccentric_sphere = resample_to_img(eccentric_sphere, sphere_mask)

    # Define the path to the new BIDS directory for the subject
    subj_dir = os.path.join(out_dir, f"sub-{subject}", "func")
    
    # Check if the directory exists; if not, make it
    if not os.path.isdir(subj_dir):
        os.makedirs(subj_dir)
    
    # Define the filename for the new Nifti file
    filename = f"sub-{subject}_task-seed_space-MNI152NLin2009cAsym_desc-coordinate{'neg' if x < 0 else 'pos'}{abs(x)}{'neg' if y < 0 else 'pos'}{abs(y)}{'neg' if z < 0 else 'pos'}{abs(z)}_bold.nii.gz"
    
    # Define the path to the new Nifti file
    save_path = os.path.join(subj_dir, filename)
    
    print('saved to: ', save_path)
    
    # Save the new Nifti file
    resampled_eccentric_sphere.to_filename(save_path)
    
    return save_path


def generate_spherical_roi(x, y, z, out_dir=rf'/Users/cu135/Dropbox (Partners HealthCare)/memory/functional_networks/seeds/spherical_seeds/misc/', radius=5):
    sphere_roi = gen_sphere_roi(xcoord=x, ycoord=y, zcoord=z, mask=False, thresh_mx=None, radius=radius);
    mask = nimds.get_img("mni_icbm152")
    ovr_img3 = image.new_img_like(mask, sphere_roi)
    #Save
    if os.path.isdir(out_dir)==False:
        os.makedirs(out_dir)
    #Save
    if x < 0:
        xsign = 'neg'
    else:
        xsign = 'pos'
    if y < 0:
        ysign = 'neg'
    else:
        ysign = 'pos'
    if z < 0:
        zsign = 'neg'
    else:
        zsign = 'pos'
    save_dir = out_dir + f'x_{xsign}{abs(x)}_y_{ysign}{abs(y)}_z_{zsign}{abs(z)}.nii'
    print('saved to: ', save_dir)
    ovr_img3.to_filename(save_dir)
    return save_dir

def gen_sphere_roi(xcoord, ycoord, zcoord, mask=False, thresh_mx=None, radius=5):
    #Prepare matrices for masking
    from nltools.mask import create_sphere
    sphere_mask = create_sphere([xcoord, ycoord, zcoord], radius=radius)
    sphere_mask = sphere_mask.get_fdata()
    print(np.count_nonzero(sphere_mask))
    if mask:
        print(np.count_nonzero(thresh_mx))
        thresh_mx[sphere_mask == 0] = 0# != sphere_mask] = 0
        print(np.count_nonzero(thresh_mx))
        out_mx = thresh_mx
        print(np.count_nonzero(out_mx))
        print('I will mask to the region around the sphere')
    else:
        out_mx = sphere_mask
        print('I will binarize wihtin the sphere')
    return out_mx

def read_coordinates_csv(filename, radius=1):
    # from calvin_utils.generate_nifti import generate_spherical_roi
    csv_df = pd.read_csv(filename)
    out_dir = os.path.join(os.path.dirname(filename), 'generated_spherical_rois/')
    coordinate_list = []
    for i in range(0, len(csv_df.index)):
        coordinates = csv_df.iloc[i, :]
        print(f'Assessing coordinates: x={coordinates[0]}, y={coordinates[1]}, z={coordinates[2]}')
        coordinate_name = generate_spherical_roi(coordinates[0], coordinates[1], coordinates[2], out_dir=out_dir, radius=radius)
        coordinate_list.append(coordinate_name)
    coordinate_df = pd.DataFrame(coordinate_list)
    return coordinate_df
        
def read_subject_coordinates_csv(filename, radius=5, eccentricity=None, method='binary'):
    # Load the CSV file into a DataFrame
    csv_df = pd.read_csv(filename)
    
    # Define the path to the directory where the BIDS dataset will be saved
    bids_dir = os.path.join(os.path.dirname(filename), 'BIDS_dataset')
    
    # Initialize an empty list to store the paths to the spherical ROIs
    coordinate_list = []
    
    # For each row in the DataFrame
    for _, row in csv_df.iterrows():
        print(f'Assessing coordinates: x={row.x}, y={row.y}, z={row.z} for subject: {row.subject}')
        
        # Create a BIDS directory for the subject
        subject_bids_dir = os.path.join(bids_dir, f"sub-{row.subject}")
        os.makedirs(subject_bids_dir, exist_ok=True)
        
        # Generate a ROI for the coordinates and add the path to the list
        if method=='concentric':
            coordinate_path = generate_concentric_spherical_roi(row.subject, row.x, row.y, row.z, out_dir=subject_bids_dir, max_radius=radius)
        elif method=='eccentric':
            coordinate_path = generate_eccentric_spherical_roi(row.x, row.y, row.z, out_dir=subject_bids_dir, radius=radius, eccentricity=eccentricity)
        elif method=='binary':
            coordinate_path = generate_spherical_roi(row.x, row.y, row.z, out_dir=subject_bids_dir, radius=radius)
        else:
            pass
        
        coordinate_list.append(coordinate_path)
    
    # Convert the list of paths to a DataFrame and return it
    return pd.DataFrame(coordinate_list, columns=['coordinate_path'])
        
def add_matrices_together(folder):
    matrices_df = import_matrices_from_folder(folder, file_pattern='/*.nii*')
    
    summed_matrix = matrices_df.sum(axis=1)
    return summed_matrix

def view_and_save_nifti(matrix, out_dir, output_name=None, silent=False):
    img = nifti_from_matrix(matrix, output_file=out_dir, output_name=output_name, silent=silent)
    if silent:
        return None
    else:
        # mask = nimds.get_img("mni_icbm152")
        ovr_html1 = plotting.view_img(img, cut_coords=(0,0,0), black_bg=False, opacity=.75, cmap='ocean_hot')
        return ovr_html1

def merge_niftis_in_folder(folder):
    added_matrces = add_matrices_together(folder)
    out_dir = os.path.join(folder, 'merged_niftis')
    view_and_save_nifti(added_matrces, out_dir=out_dir)
    
def threshold_matrix_by_another(matrix_file_1, matrix_file_2, method='under_threshold', threshold=0.05):
    '''
    This function will calculate a threshold matrix by another matrix using the method described.
    For example, if method is less_than, the threshold matrix will be all points in matrix_1 less than matrix_2
    '''
    out_dir = os.path.join(matrix_file_1.split('.')[0], f'thresholded_by_{os.path.basename(matrix_file_2).split(".")[0]}_{method}_{threshold}')
    
    matrix_df_1 = import_matrices_from_folder(matrix_file_1, file_pattern='')
    print(np.max(matrix_df_1))
    matrix_df_2 = import_matrices_from_folder(matrix_file_2, file_pattern='')
    print(np.max(matrix_df_2))
    
    if method=='less_than':
        thresholded_df = np.where(matrix_df_1<matrix_df_2, matrix_df_1, 0)
    elif method=='greater_than':
        thresholded_df = np.where(matrix_df_1>matrix_df_2, matrix_df_1, 0)
    elif method=='under_threshold':
        thresholded_df = np.where(matrix_df_2<threshold, matrix_df_1, 0)
        print('Thresholding matrix 1 by matrix 2 under threshold ', threshold)
    elif method=='over_threshold':
        thresholded_df = np.where(matrix_df_2>threshold, matrix_df_1, 0)
        print('Thresholding matrix 1 by matrix 2 over threshold ', threshold)
    else:
        print('Unknown method, please try again.')
    nifti_img = view_and_save_nifti(thresholded_df, out_dir=out_dir)
    return nifti_img

class NiftiDataFrameHandler:
    def __init__(self, df, out_dir=None, use_colnames_as_paths=False):
        """
        Initialize the NiftiDataFrameHandler class.
        
        Args:
            df (DataFrame): DataFrame containing flattened NIfTI files in each column.
            out_dir (str, optional): The directory where to save the new NIfTI files. Ignored if use_colnames_as_paths is True.
            use_colnames_as_paths (bool, optional): If True, uses column names as absolute paths for saving NIfTI files.
        """
        self.df = df
        self.out_dir = out_dir
        self.use_colnames_as_paths = use_colnames_as_paths
    
    def sanitize_paths(self):
        """
        Sanitize column names to be used as file paths.
        """
        sanitized_colnames = []
        for col_name in self.df.columns:
            # Replace '/' with '_' to prevent path errors
            sanitized_name = col_name.replace('/', '_')
            
            # Remove all non-alphanumeric and special characters except for '/'
            sanitized_name = re.sub(r'[^\w/.-]', '', sanitized_name)
            
            # Append '.nii' extension if not present
            if not sanitized_name.endswith('.nii'):
                sanitized_name += '.nii'
            
            sanitized_colnames.append(sanitized_name)
        
        self.df.columns = sanitized_colnames
    
    def create_directories(self):
        """
        Check for and create directories where NIfTI files will be saved.
        """
        if self.use_colnames_as_paths:
            for col_name in self.df.columns:
                output_dir = os.path.dirname(col_name)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
        else:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

    def save_niftis(self):
        """
        Save NIfTI files and print their locations.
        """
        for col_name in self.df.columns:
            matrix = self.df[col_name].values
            if self.use_colnames_as_paths:
                output_name = os.path.basename(col_name)
                output_dir = os.path.dirname(col_name)
            else:
                output_name = col_name
                output_dir = self.out_dir

            view_and_save_nifti(matrix, out_dir=output_dir, output_name=output_name)
            print(f"NIfTI file saved at: {os.path.join(output_dir, output_name)}")

    def run(self):
        """
        Run the entire pipeline.
        """
        self.sanitize_paths()
        self.create_directories()
        self.save_niftis()
