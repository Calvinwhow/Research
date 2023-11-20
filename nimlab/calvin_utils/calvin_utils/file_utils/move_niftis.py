import os
import shutil
import glob

def find_and_move_nifti_files(source_pattern, destination_folder, dry_run=True):
    """
    Find all NIfTI files matching the given pattern and optionally move them to a specified folder.

    Parameters:
    - source_pattern: The file pattern to search for (e.g., '/path/to/files/*/*.nii').
    - destination_folder: The folder to move the files to.
    - dry_run: If True, only print the filenames; if False, move the files.
    """
    # Find all files matching the pattern
    nifti_files = glob.glob(source_pattern)
    
    for file in nifti_files:
        # Print the file path for dry run
        print(f"Found file: {file}")
        
        if not dry_run:
            # Move the file to the destination folder
            destination_path = os.path.join(destination_folder, os.path.basename(file))
            shutil.move(file, destination_path)
            print(f"Moved file to: {destination_path}")


if __name__=='__main__':
    # Set your variables
    NIFTI_PATH = "/data/nimlab/dl_archive/adni_calvin/raws/*/*/*.nii"
    DESTINATION_FOLDER = "/data/nimlab/dl_archive/adni_calvin/raws/all_pts"  # Replace with your actual destination folder path
    DRY_RUN=True
    # Example usage
    find_and_move_nifti_files(NIFTI_PATH, DESTINATION_FOLDER, dry_run=DRY_RUN)  # Dry run