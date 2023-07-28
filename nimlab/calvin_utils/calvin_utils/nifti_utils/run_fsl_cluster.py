import os
import subprocess

def run_fsl_cluster(nifti_file_path, outdir, threshold=0.1, min_cluster_size=10):
    """
    Runs FSL Cluster on a NIFTI file and saves the output to a directory.

    Parameters:
    nifti_file_path (str): Path to the NIFTI file.
    outdir (str): Output directory for the FSL Cluster results.
    threshold (float): Cluster-forming threshold (default is 2.3).
    min_cluster_size (int): Minimum cluster size in voxels (default is 10).

    Returns:
    str: Path to the FSL Cluster output file.
    """
    # Check if output directory exists, if not create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Set up FSL Cluster command
    cmd = ['cluster', '-i', nifti_file_path, '-t', str(threshold), '--omax=cluster_max > ', str(os.path.join(outdir, "cluster_output.txt"))]

    # Run FSL Cluster using subprocess
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if there was an error
    if result.returncode != 0:
        error_message = result.stderr.decode('utf-8').strip()
        raise RuntimeError(f"FSL Cluster failed with error: {error_message}")

