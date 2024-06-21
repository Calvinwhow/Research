import os
import sys
import subprocess
"""
This script is designed to facilitate the execution of various bash scripts inside a Docker container for Nifti file processing. 
It automates the process of setting up the necessary Docker command with appropriate volume mounts, 
allowing users to focus on their research rather than configuration details.

Usage:
    python run_docker.py container_choice <path_to_nifti_directory> [script_directory] [script_choice]

Parameters:
    path_to_nifti_directory (str): The full path to the directory that contains the Nifti files you want to process.
                                  This directory will be mounted to the Docker container, and the specified bash script will be executed on the Nifti files in this directory.
    script_directory (str, optional): The path to the directory containing the bash script you want to run. 
                                      Defaults to 'nimlab/sbm/scripts'.
    script_choice (str, optional): The filename with extension of the script to run inside the Docker container. 
                                   Defaults to 'easy_reg.sh'.

Details:
    The script mounts two volumes to the Docker container:
        1. The directory containing your specified bash script.
        2. The directory containing your Nifti files, specified by the user when running the script.

    The Docker container used is tagged as 'lab:latest'. Ensure that this container has all the necessary dependencies installed to run the specified bash script.

Notes:
    - Ensure that Docker is installed and running on your system before executing this script.
    - This script is intended for use on Windows systems due to the path format in the example, but it can be adapted for use on other systems by adjusting the path formats.
    - Modify the script_directory and script_choice parameters as needed to suit your specific setup.
"""

def run_docker(container: str, nifti_directory: str, script_dir: str, script_choice: str):
    """
    Runs the easy_reg.sh script inside a Docker container.

    Parameters:
        container (str): The containter to use. Options: freesurfer | fsl
        nifti_directory (str): The path to the directory containing the Nifti files.
        script_dir (str): The path to the bash scripts directory you want run by the dockerfile.
        script_choice (str): The filename with extension of the script to run.
    """
    # Process file paths
    host_script_dir = script_dir.replace("\\", "/")
    host_data_dir = nifti_directory.replace("\\", "/")

    command = [
        "docker", "run", 
        "-m", "12g", "--memory-swap", "12g",
        "-v", f"{host_script_dir}:/scripts",
        "-v", f"{host_data_dir}:/data",
        f"{container}:latest",
        f"/scripts/{script_choice}"
    ]
    print("RUNNING COMMAND: \n" + " ".join(command))
    # Run the Docker command
    subprocess.run(command)

if __name__ == "__main__":
    # Check if enough arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python run_docker_easyreg.py <path_to_nifti_directory> [path_to_nifti_directory] [script_choice.sh]")
        sys.exit(1)
    
    # Assign the container (program) to use from command line arguments
    container = sys.argv[1]

    # Assign the Nifti directory from command line arguments
    nifti_directory = sys.argv[2]

    # Optional script directory and script choice, with defaults
    script_dir = sys.argv[3] if len(sys.argv) > 2 else 'nimlab/sbm/scripts'
    script_choice = sys.argv[4] if len(sys.argv) > 3 else 'easy_reg.sh'

    # Run the Docker command with the provided arguments
    run_docker(container, nifti_directory, script_dir, script_choice)

