from glob import glob
import subprocess
import shutil
import os 

from ..nifti_utils.generate_nifti import view_and_save_nifti
from ..file_utils.import_functions import GiiNiiFileImport

# User-defined variables
root_dir = r"/Users/cu135/Dropbox (Partners HealthCare)/resources/datasets/Queensland_PD_DBS_STN/BIDSdata/derivatives/leaddbs"
file_pattern_1 = r"/stimulations/MNI152NLin2009bAsym/gs_2023Aysu/*sim-binary_model-simbio_hemi-R.nii"
file_pattern_2 = r"/stimulations/MNI152NLin2009bAsym/gs_2023Aysu/*sim-binary_model-simbio_hemi-L.nii"
output_file_pattern = r"sim-efield_model-simbio_hemi-bl.nii"

# Path to MNI template
mni_template_path = r"/Users/cu135/Dropbox (Partners HealthCare)/resources/mni_spaces/mni_icbm152_nlin_sym_09b_nifti/mni_icbm152_nlin_sym_09b/mni_icbm152_t1_tal_nlin_sym_09b_hires.nii"

for sub_dir in os.listdir(root_dir):
    sub_dir_path = os.path.join(root_dir, sub_dir)
    if not os.path.isdir(sub_dir_path):
        continue
    # print("Subdirpath:", os.path.join(root_dir, sub_dir))
    files1 = glob(root_dir+"/"+sub_dir+file_pattern_1)
    files2 = glob(root_dir+"/"+sub_dir+file_pattern_2)

    if files1 and files2:
        # Extract identified files
        file1 = files1[0]
        file2 = files2[0]
        
        # Prepare Output File
        output_dir = os.path.dirname(file1)
        output_file_with_subject = os.path.basename(sub_dir_path)+"_"+output_file_pattern
        
        # Prepare intermediary file
        copyfile1 = file1.split('.nii')[0] + "_copy.nii"
        copyfile2 = file2.split('.nii')[0] + "_copy.nii"
        
        try:
            # Copy files to create backups
            # shutil.copyfile(file1, copyfile1)
            # shutil.copyfile(file2, copyfile2)

            # Attempt resampling to MNI space to fix bad headers
            # subprocess.run(["flirt", "-in", copyfile1, "-ref", mni_template_path, "-out", copyfile1], check=True)
            # subprocess.run(["flirt", "-in", copyfile2, "-ref", mni_template_path, "-out", copyfile2], check=True)

            # Add the aligned files
            importer = GiiNiiFileImport(os.path.dirname(copyfile1), file_pattern="*sim-binary_model-simbio_hemi-*copy*")
            imports = importer.run()
            display(imports)
            imports["summated"] = imports.iloc[:,0] + imports.iloc[:,1]
            
            view_and_save_nifti(imports["summated"], out_dir=output_dir, output_file=output_file_with_subject)
            
            # Clean up and remove the copy files. 
            # os.remove(copyfile1)
            # os.remove(copyfile2)
            
            print(f"\n ***Processed: {sub_dir_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing {sub_dir_path}: {e}")
    else:
        print(f"Missing file(s) in {sub_dir_path}")