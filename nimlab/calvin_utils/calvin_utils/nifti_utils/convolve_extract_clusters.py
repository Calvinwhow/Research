#Identify the clusters using convolution
import scipy.ndimage as ndimage 
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from nimlab import datasets as nimds
from nilearn import image, plotting, maskers

#----------------------------------------------------------------user input----------------------------------------------------------------
def convolve_extract_clusters(nifti_file, save_clusters=True):    
    nifti_image = image.load_img(nifti_file)
    conjunction_data = nifti_image.get_fdata()

    kernel = np.ones((3,3,3))
    binary_mask = conjunction_data
    convolution = []
    for x in range(0, binary_mask.shape[0]): #iterate through all the x xalues
        for y in range(0, binary_mask.shape[1]): #iterate through all the y values
            for z in range(0, binary_mask.shape[2]): #iterate through all the z values
                try: #If there are 9 voxels, do this
                    # Perform convolution with the binary mask and the kernel, appending to array.
                    convolution.append(np.squeeze(np.array([
                        np.sum(np.multiply(binary_mask[x:x+3, y:y+3, z:z+3], kernel))
                    ])))
                except: # if there are <9 voxels, do this. 
                    convolution.append(0)
    # Reshape the convolved array to match the shape of the original data
    convolution = np.array(convolution)
    c, numc = ndimage.measurements.label(convolution)
    convolution = convolution.reshape(binary_mask.shape[0], binary_mask.shape[1], binary_mask.shape[2])
    c, numc = ndimage.measurements.label(convolution)
    print('Number clusters: ', numc)

    mask = nimds.get_img("mni_icbm152")
    cluster_dict = {}
    for i in range(1, numc+1):
        cluster_dict[i] = np.where(c == i, 1, 0)
        print(f'Cluster {i} is size: {np.sum(cluster_dict[i])}')
        print(cluster_dict[i].shape)
        if save_clusters:
            cluster_out = nifti_file.split('.nii')[0] + '_clustered_rois'
            savename = 'cluster_' + str(i)
            
            if os.path.isdir(cluster_out) != True:
                os.makedirs(cluster_out)
            
            cluster_img = image.new_img_like(mask, cluster_dict[i]);
            cluster_html = plotting.view_img(cluster_img, cut_coords=(0,0,0), title=(f'cluster_{i}'), black_bg=False, opacity=.75, cmap='ocean_hot');
            cluster_img.to_filename(os.path.join(cluster_out, f'{savename}'));
            cluster_html.save_as_html(os.path.join(cluster_out, f'{savename}.html'));

            print('File: ' + savename)
            print('saved to: ', cluster_out)
        # cluster_matrix = cluster.reshape(-1, np.prod(cluster.shape)).T
    return cluster_dict