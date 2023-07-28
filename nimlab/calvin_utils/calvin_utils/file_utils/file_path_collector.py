#NOTEBOOK TO CREATE CSVS OF ALL DESIRED FILE PATHS
from glob import glob
import os
import pandas as pd
from pathlib import Path

##------USER INPUT BELOW-----
root_dir = '/PHShome/cu135/memory/connectome_inputs/bulks_for_roi_roi_correl/bulk_seven'
#This is the directory from which all relevant files are below
inter_dirs = '/*'
file_pattern = '.nii*' #this is the common naming convention relevant to all files 

##------END USER INPUT-------

name = file_pattern.split('.')[0]
file_name = f'paths_to_{os.path.basename(root_dir)}_{name}.csv'
print(file_name)
column_title = file_name
print('Will save as:', file_name)
out_dir = '/PHShome/cu135/memory/file_paths' #where do you want your files to go 

glob_path = root_dir + inter_dirs + file_pattern
print(glob_path)
glob_list = glob(glob_path)
csv_df = pd.DataFrame({'paths': glob_list}) #produces nested lists of the csvs

if os.path.exists(out_dir):
    pass
else:
    os.mkdir(out_dir)
    
csv_df.to_csv(os.path.join(out_dir,file_name), header=False, index=False)
print('saved to:', out_dir, 'as \n', file_name)