#NOTEBOOK TO CREATE CSVS OF ALL DESIRED FILE PATHS
from glob import glob
import os
import pandas as pd
from pathlib import Path

def glob_file_paths(shared_base_path, shared_file_pattern='', save=False):
    glob_path  = os.path.join(shared_base_path, shared_file_pattern)
    
    print('I will search: ', glob_path)

    globbed = glob(glob_path)
    #Identify files of interest
    path_df = pd.DataFrame({})
    try:
        path_df['paths'] = globbed
    except:
        path_df = pd.DataFrame({'paths': globbed})
    if save:
        save_path_df(path_df, out_dir=shared_base_path)
    return path_df

def save_path_df(path_df, out_dir):
    path_df.to_csv(os.path.join(out_dir, 'path_df.csv'))
    print('Saved path: ', os.path.join(out_dir, 'path_df.csv'))

if __name__=='__main__':
    ##------USER INPUT BELOW-----
    root_dir = '/PHShome/cu135/memory/connectome_inputs/bulks_for_roi_roi_correl/bulk_seven'
    #This is the directory from which all relevant files are below
    inter_dirs = '/*'
    file_pattern = '.nii*' #this is the common naming convention relevant to all files 
    path_df = glob_file_paths(root_dir, file_pattern)